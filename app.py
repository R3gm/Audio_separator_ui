import os
import spaces
import gc
import hashlib
import queue
import threading
import json
import shlex
import sys
import subprocess
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from utils import (
    remove_directory_contents,
    create_directories,
    download_manager,
)
import random
from utils import logger
import onnxruntime as ort
import warnings
import gradio as gr
import time
import traceback
from pedalboard import Pedalboard, Reverb, Delay, Chorus, Compressor, Gain, HighpassFilter, LowpassFilter
from pedalboard.io import AudioFile
import argparse

parser = argparse.ArgumentParser(description="Run the app with optional sharing")
parser.add_argument(
    '--share',
    action='store_true',
    help='Enable sharing mode'
)
parser.add_argument(
    '--theme',
    type=str,
    default="NoCrypt/miku",
    help='Set the theme (default: NoCrypt/miku)'
)
args = parser.parse_args()

warnings.filterwarnings("ignore")
IS_COLAB = True if ('google.colab' in sys.modules or args.share) else False
IS_ZERO_GPU = os.getenv("SPACES_ZERO_GPU")

title = "<center><strong><font size='7'>Audio🔹separator</font></strong></center>"
base_demo = "This demo uses the "
description = (f"{base_demo if IS_ZERO_GPU else ''}MDX-Net models for vocal and background sound separation.")
RESOURCES = "- You can also try `Audio🔹separator` in Colab’s free tier, which provides free GPU [link](https://github.com/R3gm/Audio_separator_ui?tab=readme-ov-file#audio-separator)."
theme = args.theme

stem_naming = {
    "Vocals": "Instrumental",
    "Other": "Instruments",
    "Instrumental": "Vocals",
    "Drums": "Drumless",
    "Bass": "Bassless",
}


class MDXModel:
    def __init__(
        self,
        device,
        dim_f,
        dim_t,
        n_fft,
        hop=1024,
        stem_name=None,
        compensation=1.000,
    ):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(
            window_length=self.n_fft, periodic=True
        ).to(device)

        out_c = self.dim_c

        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 4, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        # c = 4*2 if self.target_name=='*' else 2
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
        )
        return x.reshape([-1, 2, self.chunk_size])


class MDX:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(
        self, model_path: str, params: MDXModel, processor=0
    ):
        # Set the device and the provider (CPU or CUDA)
        self.device = (
            torch.device(f"cuda:{processor}")
            if processor >= 0
            else torch.device("cpu")
        )
        self.provider = (
            ["CUDAExecutionProvider"]
            if processor >= 0
            else ["CPUExecutionProvider"]
        )

        self.model = params

        # Load the ONNX model using ONNX Runtime
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        # Preload the model for faster performance
        self.ort.run(
            None,
            {"input": torch.rand(1, 4, params.dim_f, params.dim_t).numpy()},
        )
        self.process = lambda spec: self.ort.run(
            None, {"input": spec.cpu().numpy()}
        )[0]

        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except: # noqa
            model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()

        return model_hash

    @staticmethod
    def segment(
        wave,
        combine=True,
        chunk_size=DEFAULT_CHUNK_SIZE,
        margin_size=DEFAULT_MARGIN_SIZE,
    ):
        """
        Segment or join segmented wave array

        Args:
            wave: (np.array) Wave array to be segmented or joined
            combine: (bool) If True, combines segmented wave array.
                If False, segments wave array.
            chunk_size: (int) Size of each segment (in samples)
            margin_size: (int) Size of margin between segments (in samples)

        Returns:
            numpy array: Segmented or joined wave array
        """

        if combine:
            # Initializing as None instead of [] for later numpy array concatenation
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:  # Create array for first segment
                    processed_wave = segment[:, start:end]
                else:  # Concatenate to existing array for subsequent segments
                    processed_wave = np.concatenate(
                        (processed_wave, segment[:, start:end]), axis=-1
                    )

        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(
                range(0, sample_count, chunk_size)
            ):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        """
        Pad the wave array to match the required chunk size

        Args:
            wave: (np.array) Wave array to be padded

        Returns:
            tuple: (padded_wave, pad, trim)
                - padded_wave: Padded wave array
                - pad: Number of samples that were padded
                - trim: Number of samples that were trimmed
        """
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        # Padded wave
        wave_p = np.concatenate(
            (
                np.zeros((2, trim)),
                wave,
                np.zeros((2, pad)),
                np.zeros((2, trim)),
            ),
            1,
        )

        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(
            self.device
        )

        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        """
        Process each wave segment in a multi-threaded environment

        Args:
            mix_waves: (torch.Tensor) Wave segments to be processed
            trim: (int) Number of samples trimmed during padding
            pad: (int) Number of samples padded during padding
            q: (queue.Queue) Queue to hold the processed wave segments
            _id: (int) Identifier of the processed wave segment

        Returns:
            numpy array: Processed wave segment
        """
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(
                    processed_spec.to(self.device)
                )
                processed_wav = (
                    processed_wav[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()
                    .numpy()
                )
                pw.append(processed_wav)
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        q.put({_id: processed_signal})
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        """
        Process the wave array in a multi-threaded environment

        Args:
            wave: (np.array) Wave array to be processed
            mt_threads: (int) Number of threads to be used for processing

        Returns:
            numpy array: Processed wave array
        """
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)

        # Create a queue to hold the processed wave segments
        q = queue.Queue()
        threads = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            self.prog.total = len(mix_waves) * mt_threads
            thread = threading.Thread(
                target=self._process_wave, args=(mix_waves, trim, pad, q, c)
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.prog.close()

        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [
            list(wave.values())[0]
            for wave in sorted(
                processed_batches, key=lambda d: list(d.keys())[0]
            )
        ]
        assert len(processed_batches) == len(
            waves
        ), "Incomplete processed batches, please reduce batch size!"
        return self.segment(processed_batches, True, chunk)


@spaces.GPU(duration=30)
def run_mdx(
    model_params,
    output_dir,
    model_path,
    filename,
    exclude_main=False,
    exclude_inversion=False,
    suffix=None,
    invert_suffix=None,
    denoise=False,
    keep_orig=True,
    m_threads=2,
    device_base="cuda",
):

    if device_base == "cuda":
        device = torch.device("cuda:0")
        processor_num = 0
        device_properties = torch.cuda.get_device_properties(device)
        vram_gb = device_properties.total_memory / 1024**3
        m_threads = 1 if vram_gb < 8 else (8 if vram_gb > 32 else 2)
        duration = librosa.get_duration(filename=filename)
        if duration < 60:
            m_threads = 1
        logger.info(f"threads: {m_threads} vram: {vram_gb}")
    else:
        device = torch.device("cpu")
        processor_num = -1
        m_threads = 1

    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"],
    )

    mdx_sess = MDX(model_path, model, processor=processor_num)
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    # normalizing input wave gives better output
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak
    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (
            mdx_sess.process_wave(wave, m_threads)
        )
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)
    # return to previous peak
    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = (
            stem_naming.get(stem_name)
            if invert_suffix is None
            else invert_suffix
        )
        stem_name = (
            f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        )
        invert_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(
            invert_filepath,
            (-wave_processed.T * model.compensation) + wave.T,
            sr,
        )

    if not keep_orig:
        os.remove(filename)

    del mdx_sess, wave_processed, wave
    gc.collect()
    torch.cuda.empty_cache()
    return main_filepath, invert_filepath


def run_mdx_beta(
    model_params,
    output_dir,
    model_path,
    filename,
    exclude_main=False,
    exclude_inversion=False,
    suffix=None,
    invert_suffix=None,
    denoise=False,
    keep_orig=True,
    m_threads=2,
    device_base="",
):

    m_threads = 1
    duration = librosa.get_duration(filename=filename)
    if IS_COLAB or duration < 60:
        m_threads = 1
    elif duration >= 60 and duration <= 120:
        m_threads = 8
    elif duration > 120:
        m_threads = 16

    logger.info(f"threads: {m_threads}")

    model_hash = MDX.get_hash(model_path)
    device = torch.device("cpu")
    processor_num = -1
    mp = model_params.get(model_hash)
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"],
    )

    mdx_sess = MDX(model_path, model, processor=processor_num)
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    # normalizing input wave gives better output
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak
    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (
            mdx_sess.process_wave(wave, m_threads)
        )
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)
    # return to previous peak
    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = (
            stem_naming.get(stem_name)
            if invert_suffix is None
            else invert_suffix
        )
        stem_name = (
            f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        )
        invert_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(
            invert_filepath,
            (-wave_processed.T * model.compensation) + wave.T,
            sr,
        )

    if not keep_orig:
        os.remove(filename)

    del mdx_sess, wave_processed, wave
    gc.collect()
    torch.cuda.empty_cache()
    return main_filepath, invert_filepath


MDX_DOWNLOAD_LINK = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
UVR_MODELS = [
    "UVR-MDX-NET-Voc_FT.onnx",
    "UVR_MDXNET_KARA_2.onnx",
    "Reverb_HQ_By_FoxJoy.onnx",
    "UVR-MDX-NET-Inst_HQ_4.onnx",
]
BASE_DIR = "."  # os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdxnet_models_dir = os.path.join(BASE_DIR, "mdx_models")
output_dir = os.path.join(BASE_DIR, "clean_song_output")


def convert_to_stereo_and_wav(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    # check if mono
    if type(wave[0]) != np.ndarray or audio_path[-4:].lower() != ".wav": # noqa
        stereo_path = f"{os.path.splitext(audio_path)[0]}_stereo.wav"
        stereo_path = os.path.join(output_dir, stereo_path)

        command = shlex.split(
            f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"'
        )
        sub_params = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "creationflags": subprocess.CREATE_NO_WINDOW
            if sys.platform == "win32"
            else 0,
        }
        process_wav = subprocess.Popen(command, **sub_params)
        output, errors = process_wav.communicate()
        if process_wav.returncode != 0 or not os.path.exists(stereo_path):
            raise Exception("Error processing audio to stereo wav")

        return stereo_path
    else:
        return audio_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:18]


def random_sleep():
    sleep_time = 0.1
    if IS_ZERO_GPU:
        sleep_time = round(random.uniform(3.2, 5.9), 1)
    time.sleep(sleep_time)


def process_uvr_task(
    orig_song_path: str = "aud_test.mp3",
    main_vocals: bool = False,
    dereverb: bool = True,
    song_id: str = "mdx",  # folder output name
    only_voiceless: bool = False,
    remove_files_output_dir: bool = False,
):

    device_base = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_base}")

    if remove_files_output_dir:
        remove_directory_contents(output_dir)

    with open(os.path.join(mdxnet_models_dir, "data.json")) as infile:
        mdx_model_params = json.load(infile)

    song_output_dir = os.path.join(output_dir, song_id)
    create_directories(song_output_dir)
    orig_song_path = convert_to_stereo_and_wav(orig_song_path)

    logger.info(f"onnxruntime device >> {ort.get_device()}")

    if only_voiceless:
        logger.info("Voiceless Track Separation...")

        process = run_mdx(
            mdx_model_params,
            song_output_dir,
            os.path.join(mdxnet_models_dir, "UVR-MDX-NET-Inst_HQ_4.onnx"),
            orig_song_path,
            suffix="Voiceless",
            denoise=False,
            keep_orig=True,
            exclude_inversion=True,
            device_base=device_base,
        )

        return process

    logger.info("Vocal Track Isolation...")
    vocals_path, instrumentals_path = run_mdx(
        mdx_model_params,
        song_output_dir,
        os.path.join(mdxnet_models_dir, "UVR-MDX-NET-Voc_FT.onnx"),
        orig_song_path,
        denoise=True,
        keep_orig=True,
        device_base=device_base,
    )

    if main_vocals:
        random_sleep()
        msg_main = "Main Voice Separation from Supporting Vocals..."
        logger.info(msg_main)
        gr.Info(msg_main)
        try:
            backup_vocals_path, main_vocals_path = run_mdx(
                mdx_model_params,
                song_output_dir,
                os.path.join(mdxnet_models_dir, "UVR_MDXNET_KARA_2.onnx"),
                vocals_path,
                suffix="Backup",
                invert_suffix="Main",
                denoise=True,
                device_base=device_base,
            )
        except Exception as e:
            backup_vocals_path, main_vocals_path = run_mdx_beta(
                mdx_model_params,
                song_output_dir,
                os.path.join(mdxnet_models_dir, "UVR_MDXNET_KARA_2.onnx"),
                vocals_path,
                suffix="Backup",
                invert_suffix="Main",
                denoise=True,
                device_base=device_base,
            )
    else:
        backup_vocals_path, main_vocals_path = None, vocals_path

    if dereverb:
        random_sleep()
        msg_dereverb = "Vocal Clarity Enhancement through De-Reverberation..."
        logger.info(msg_dereverb)
        gr.Info(msg_dereverb)
        try:
            _, vocals_dereverb_path = run_mdx(
                mdx_model_params,
                song_output_dir,
                os.path.join(mdxnet_models_dir, "Reverb_HQ_By_FoxJoy.onnx"),
                main_vocals_path,
                invert_suffix="DeReverb",
                exclude_main=True,
                denoise=True,
                device_base=device_base,
            )
        except Exception as e:
            _, vocals_dereverb_path = run_mdx_beta(
                mdx_model_params,
                song_output_dir,
                os.path.join(mdxnet_models_dir, "Reverb_HQ_By_FoxJoy.onnx"),
                main_vocals_path,
                invert_suffix="DeReverb",
                exclude_main=True,
                denoise=True,
                device_base=device_base,
            )
    else:
        vocals_dereverb_path = main_vocals_path

    return (
        vocals_path,
        instrumentals_path,
        backup_vocals_path,
        main_vocals_path,
        vocals_dereverb_path,
    )


def add_vocal_effects(input_file, output_file, reverb_room_size=0.6, vocal_reverb_dryness=0.8, reverb_damping=0.6, reverb_wet_level=0.35,
                      delay_seconds=0.4, delay_mix=0.25,
                      compressor_threshold_db=-25, compressor_ratio=3.5, compressor_attack_ms=10, compressor_release_ms=60,
                      gain_db=3):

    effects = [HighpassFilter()]

    effects.append(Reverb(room_size=reverb_room_size, damping=reverb_damping, wet_level=reverb_wet_level, dry_level=vocal_reverb_dryness))

    effects.append(Compressor(threshold_db=compressor_threshold_db, ratio=compressor_ratio,
                              attack_ms=compressor_attack_ms, release_ms=compressor_release_ms))

    if delay_seconds > 0 or delay_mix > 0:
        effects.append(Delay(delay_seconds=delay_seconds, mix=delay_mix))
        # print("delay applied")
    # effects.append(Chorus())

    if gain_db:
        effects.append(Gain(gain_db=gain_db))
        # print("added gain db")

    board = Pedalboard(effects)

    with AudioFile(input_file) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)


def add_instrumental_effects(input_file, output_file, highpass_freq=100, lowpass_freq=12000,
                             reverb_room_size=0.5, reverb_damping=0.5, reverb_wet_level=0.25,
                             compressor_threshold_db=-20, compressor_ratio=2.5, compressor_attack_ms=15, compressor_release_ms=80,
                             gain_db=2):

    effects = [
        HighpassFilter(cutoff_frequency_hz=highpass_freq),
        LowpassFilter(cutoff_frequency_hz=lowpass_freq),
    ]
    if reverb_room_size > 0 or reverb_damping > 0 or reverb_wet_level > 0:
        effects.append(Reverb(room_size=reverb_room_size, damping=reverb_damping, wet_level=reverb_wet_level))

    effects.append(Compressor(threshold_db=compressor_threshold_db, ratio=compressor_ratio,
                              attack_ms=compressor_attack_ms, release_ms=compressor_release_ms))

    if gain_db:
        effects.append(Gain(gain_db=gain_db))

    board = Pedalboard(effects)

    with AudioFile(input_file) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)


COMMON_SAMPLE_RATES = [8000, 16000, 22050, 32000, 44100, 48000, 96000]


def save_audio(audio_opt: np.ndarray, final_sr: int, output_audio_path: str, target_format: str) -> str:
    """
    Save audio with automatic handling of unsupported sample rates for non-WAV formats.
    """
    ext = os.path.splitext(output_audio_path)[1].lower()

    try:
        if ext == ".wav":
            sf.write(output_audio_path, audio_opt, final_sr, format=target_format)
        else:
            target_sr = min(COMMON_SAMPLE_RATES, key=lambda altsr: abs(altsr - final_sr))
            if target_sr != final_sr:
                logger.warning(f"Resampling from {final_sr} -> {target_sr} for {ext}")
                audio_opt = librosa.resample(audio_opt, orig_sr=final_sr, target_sr=target_sr)
            sf.write(output_audio_path, audio_opt, target_sr, format=target_format)
    except Exception as e:
        logger.error(e)
        logger.error(f"Error saving {output_audio_path}, performing fallback to WAV")
        output_audio_path = output_audio_path.replace(f"_converted.{target_format}", ".wav")

    return output_audio_path


def convert_format(file_paths, media_dir, target_format):
    """
    Convert a list of audio files to the target format with automatic safe sample rates.

    WAV files are returned as-is; non-WAV files are resampled if needed to a supported rate.
    """
    target_format = target_format.lower()
    if target_format == "wav":
        return file_paths  # No conversion needed for WAV

    suffix = "_converted"
    converted_files = []

    for fp in file_paths:
        # Absolute paths and base filename
        abs_fp = os.path.abspath(fp)
        file_name, _ = os.path.splitext(os.path.basename(abs_fp))
        file_ext = f".{target_format}"
        out_name = file_name + suffix + file_ext
        out_path = os.path.join(media_dir, out_name)

        # Load audio with librosa (handles many formats)
        audio, sr = sf.read(abs_fp)

        # Save using safe resampling
        saved_path = save_audio(audio, sr, out_path, target_format)
        converted_files.append(saved_path)

        # print(f"Converted: {abs_fp} -> {saved_path}")

    return converted_files


def sound_separate(
    media_file, stem, main, dereverb, vocal_effects=True, background_effects=True,
    vocal_reverb_room_size=0.6, vocal_reverb_damping=0.6, vocal_reverb_dryness=0.8, vocal_reverb_wet_level=0.35,
    vocal_delay_seconds=0.4, vocal_delay_mix=0.25,
    vocal_compressor_threshold_db=-25, vocal_compressor_ratio=3.5, vocal_compressor_attack_ms=10, vocal_compressor_release_ms=60,
    vocal_gain_db=4,
    background_highpass_freq=120, background_lowpass_freq=11000,
    background_reverb_room_size=0.5, background_reverb_damping=0.5, background_reverb_wet_level=0.25,
    background_compressor_threshold_db=-20, background_compressor_ratio=2.5, background_compressor_attack_ms=15, background_compressor_release_ms=80,
    background_gain_db=3,
    target_format="WAV",
):
    if not media_file:
        raise ValueError("The audio path is missing.")

    if not stem:
        raise ValueError("Please select 'vocal' or 'background' stem.")

    hash_audio = str(get_hash(media_file))
    media_dir = os.path.dirname(media_file)

    outputs = []

    try:
        duration_base_ = librosa.get_duration(filename=media_file)
        print("Duration audio:", duration_base_)
    except Exception as e:
        print(e)

    start_time = time.time()

    if "vocal" in stem:
        try:
            _, _, _, _, vocal_audio = process_uvr_task(
                orig_song_path=media_file,
                song_id=hash_audio + "mdx",
                main_vocals=main,
                dereverb=dereverb,
                remove_files_output_dir=False,
            )

            if vocal_effects:
                suffix = '_effects'
                file_name, file_extension = os.path.splitext(os.path.abspath(vocal_audio))
                out_effects = file_name + suffix + file_extension
                out_effects_path = os.path.join(media_dir, out_effects)
                add_vocal_effects(vocal_audio, out_effects_path,
                                  reverb_room_size=vocal_reverb_room_size, reverb_damping=vocal_reverb_damping, vocal_reverb_dryness=vocal_reverb_dryness, reverb_wet_level=vocal_reverb_wet_level,
                                  delay_seconds=vocal_delay_seconds, delay_mix=vocal_delay_mix,
                                  compressor_threshold_db=vocal_compressor_threshold_db, compressor_ratio=vocal_compressor_ratio, compressor_attack_ms=vocal_compressor_attack_ms, compressor_release_ms=vocal_compressor_release_ms,
                                  gain_db=vocal_gain_db
                                  )
                vocal_audio = out_effects_path

            outputs.append(vocal_audio)
        except Exception as error:
            gr.Info(str(error))
            logger.error(str(error))

    if "background" in stem:
        background_audio, _ = process_uvr_task(
            orig_song_path=media_file,
            song_id=hash_audio + "voiceless",
            only_voiceless=True,
            remove_files_output_dir=False,
        )

        if background_effects:
            suffix = '_effects'
            file_name, file_extension = os.path.splitext(os.path.abspath(background_audio))
            out_effects = file_name + suffix + file_extension
            out_effects_path = os.path.join(media_dir, out_effects)
            # print(file_name, file_extension, out_effects, out_effects_path)
            add_instrumental_effects(background_audio, out_effects_path,
                                     highpass_freq=background_highpass_freq, lowpass_freq=background_lowpass_freq,
                                     reverb_room_size=background_reverb_room_size, reverb_damping=background_reverb_damping, reverb_wet_level=background_reverb_wet_level,
                                     compressor_threshold_db=background_compressor_threshold_db, compressor_ratio=background_compressor_ratio, compressor_attack_ms=background_compressor_attack_ms, compressor_release_ms=background_compressor_release_ms,
                                     gain_db=background_gain_db
                                     )
            background_audio = out_effects_path

        outputs.append(background_audio)

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time} seconds")

    if not outputs:
        raise Exception("Error in sound separation.")

    return convert_format(outputs, media_dir, target_format)


def audio_downloader(
    url_media,
):

    url_media = url_media.strip()

    if not url_media:
        return None

    if IS_ZERO_GPU and "youtube.com" in url_media:
        gr.Info("This option isn’t available on Hugging Face.")
        return None

    import yt_dlp
    # print(url_media[:10])

    dir_output_downloads = "downloads"
    os.makedirs(dir_output_downloads, exist_ok=True)

    media_info = yt_dlp.YoutubeDL(
        {"quiet": True, "no_warnings": True, "noplaylist": True}
    ).extract_info(url_media, download=False)
    download_path = f"{os.path.join(dir_output_downloads, media_info['title'])}.m4a"

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'force_overwrites': True,
        'noplaylist': True,
        'no_warnings': True,
        'quiet': True,
        'ignore_no_formats_error': True,
        'restrictfilenames': True,
        'outtmpl': download_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
        ydl_download.download([url_media])

    return download_path


def downloader_conf():
    return gr.Checkbox(
        False,
        label="URL-to-Audio",
        # info="",
        container=False,
    )


def url_media_conf():
    return gr.Textbox(
        value="",
        label="Enter URL",
        placeholder="www.youtube.com/watch?v=g_9rPvbENUw",
        visible=False,
        lines=1,
    )


def url_button_conf():
    return gr.Button(
        "Go",
        variant="secondary",
        visible=False,
    )


def show_components_downloader(value_active):
    return gr.update(
        visible=value_active
    ), gr.update(
        visible=value_active
    )


def audio_conf():
    return gr.File(
        label="Audio file",
        # file_count="multiple",
        type="filepath",
        container=True,
    )


def stem_conf():
    return gr.CheckboxGroup(
        choices=["vocal", "background"],
        value="vocal",
        label="Stem",
        # info="",
    )


def main_conf():
    return gr.Checkbox(
        False,
        label="Main",
        # info="",
    )


def dereverb_conf():
    return gr.Checkbox(
        False,
        label="Dereverb",
        # info="",
        visible=True,
    )


def vocal_effects_conf():
    return gr.Checkbox(
        False,
        label="Vocal Effects",
        # info="",
        visible=True,
    )


def background_effects_conf():
    return gr.Checkbox(
        False,
        label="Background Effects",
        # info="",
        visible=False,
    )


def vocal_reverb_room_size_conf():
    return gr.Number(
        0.15,
        label="Vocal Reverb Room Size",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        visible=True,
    )


def vocal_reverb_damping_conf():
    return gr.Number(
        0.7,
        label="Vocal Reverb Damping",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        visible=True,
    )


def vocal_reverb_wet_level_conf():
    return gr.Number(
        0.2,
        label="Vocal Reverb Wet Level",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        visible=True,
    )


def vocal_reverb_dryness_level_conf():
    return gr.Number(
        0.8,
        label="Vocal Reverb Dryness Level",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        visible=True,
    )


def vocal_delay_seconds_conf():
    return gr.Number(
        0.,
        label="Vocal Delay Seconds",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        visible=True,
    )


def vocal_delay_mix_conf():
    return gr.Number(
        0.,
        label="Vocal Delay Mix",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        visible=True,
    )


def vocal_compressor_threshold_db_conf():
    return gr.Number(
        -15,
        label="Vocal Compressor Threshold (dB)",
        minimum=-60,
        maximum=0,
        step=1,
        visible=True,
    )


def vocal_compressor_ratio_conf():
    return gr.Number(
        4.,
        label="Vocal Compressor Ratio",
        minimum=0,
        maximum=20,
        step=0.1,
        visible=True,
    )


def vocal_compressor_attack_ms_conf():
    return gr.Number(
        1.0,
        label="Vocal Compressor Attack (ms)",
        minimum=0,
        maximum=1000,
        step=1,
        visible=True,
    )


def vocal_compressor_release_ms_conf():
    return gr.Number(
        100,
        label="Vocal Compressor Release (ms)",
        minimum=0,
        maximum=3000,
        step=1,
        visible=True,
    )


def vocal_gain_db_conf():
    return gr.Number(
        0,
        label="Vocal Gain (dB)",
        minimum=-40,
        maximum=40,
        step=1,
        visible=True,
    )


def background_highpass_freq_conf():
    return gr.Number(
        120,
        label="Background Highpass Frequency (Hz)",
        minimum=0,
        maximum=1000,
        step=1,
        visible=True,
    )


def background_lowpass_freq_conf():
    return gr.Number(
        11000,
        label="Background Lowpass Frequency (Hz)",
        minimum=0,
        maximum=20000,
        step=1,
        visible=True,
    )


def background_reverb_room_size_conf():
    return gr.Number(
        0.1,
        label="Background Reverb Room Size",
        minimum=0.0,
        maximum=1.0,
        step=0.1,
        visible=True,
    )


def background_reverb_damping_conf():
    return gr.Number(
        0.5,
        label="Background Reverb Damping",
        minimum=0.0,
        maximum=1.0,
        step=0.1,
        visible=True,
    )


def background_reverb_wet_level_conf():
    return gr.Number(
        0.25,
        label="Background Reverb Wet Level",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        visible=True,
    )


def background_compressor_threshold_db_conf():
    return gr.Number(
        -15,
        label="Background Compressor Threshold (dB)",
        minimum=-60,
        maximum=0,
        step=1,
        visible=True,
    )


def background_compressor_ratio_conf():
    return gr.Number(
        4.,
        label="Background Compressor Ratio",
        minimum=0,
        maximum=20,
        step=0.1,
        visible=True,
    )


def background_compressor_attack_ms_conf():
    return gr.Number(
        15,
        label="Background Compressor Attack (ms)",
        minimum=0,
        maximum=1000,
        step=1,
        visible=True,
    )


def background_compressor_release_ms_conf():
    return gr.Number(
        60,
        label="Background Compressor Release (ms)",
        minimum=0,
        maximum=3000,
        step=1,
        visible=True,
    )


def background_gain_db_conf():
    return gr.Number(
        0,
        label="Background Gain (dB)",
        minimum=-40,
        maximum=40,
        step=1,
        visible=True,
    )


def button_conf():
    return gr.Button(
        "Inference",
        variant="primary",
    )


def output_conf():
    return gr.File(
        label="Result",
        file_count="multiple",
        interactive=False,
    )


def show_vocal_components(value_name):
    v_ = "vocal" in value_name
    b_ = "background" in value_name

    return gr.update(visible=v_), gr.update(
        visible=v_
    ), gr.update(visible=v_), gr.update(
        visible=b_
    )


FORMAT_OPTIONS = ["WAV", "MP3", "FLAC"]


def format_conf():
    return gr.Dropdown(
        choices=FORMAT_OPTIONS,
        value=FORMAT_OPTIONS[0],
        label="Format output:"
    )


def get_gui(theme):
    with gr.Blocks(theme=theme, delete_cache=(3200, 10800)) as app:
        gr.Markdown(title)
        gr.Markdown(description)

        downloader_gui = downloader_conf()
        with gr.Row():
            with gr.Column(scale=2):
                url_media_gui = url_media_conf()
            with gr.Column(scale=1):
                url_button_gui = url_button_conf()

        downloader_gui.change(
            show_components_downloader,
            [downloader_gui],
            [url_media_gui, url_button_gui]
        )

        aud = audio_conf()

        url_button_gui.click(
            audio_downloader,
            [url_media_gui],
            [aud]
        )

        with gr.Column():
            with gr.Row():
                stem_gui = stem_conf()

        with gr.Column():
            with gr.Row():
                main_gui = main_conf()
                dereverb_gui = dereverb_conf()
                vocal_effects_gui = vocal_effects_conf()
                background_effects_gui = background_effects_conf()

            with gr.Accordion("Vocal Effects Parameters", open=False):
                with gr.Row():
                    vocal_reverb_room_size_gui = vocal_reverb_room_size_conf()
                    vocal_reverb_damping_gui = vocal_reverb_damping_conf()
                    vocal_reverb_dryness_gui = vocal_reverb_dryness_level_conf()
                    vocal_reverb_wet_level_gui = vocal_reverb_wet_level_conf()
                    vocal_delay_seconds_gui = vocal_delay_seconds_conf()
                    vocal_delay_mix_gui = vocal_delay_mix_conf()
                    vocal_compressor_threshold_db_gui = vocal_compressor_threshold_db_conf()
                    vocal_compressor_ratio_gui = vocal_compressor_ratio_conf()
                    vocal_compressor_attack_ms_gui = vocal_compressor_attack_ms_conf()
                    vocal_compressor_release_ms_gui = vocal_compressor_release_ms_conf()
                    vocal_gain_db_gui = vocal_gain_db_conf()

            with gr.Accordion("Background Effects Parameters", open=False):
                with gr.Row():
                    background_highpass_freq_gui = background_highpass_freq_conf()
                    background_lowpass_freq_gui = background_lowpass_freq_conf()
                    background_reverb_room_size_gui = background_reverb_room_size_conf()
                    background_reverb_damping_gui = background_reverb_damping_conf()
                    background_reverb_wet_level_gui = background_reverb_wet_level_conf()
                    background_compressor_threshold_db_gui = background_compressor_threshold_db_conf()
                    background_compressor_ratio_gui = background_compressor_ratio_conf()
                    background_compressor_attack_ms_gui = background_compressor_attack_ms_conf()
                    background_compressor_release_ms_gui = background_compressor_release_ms_conf()
                    background_gain_db_gui = background_gain_db_conf()

            stem_gui.change(
                show_vocal_components,
                [stem_gui],
                [main_gui, dereverb_gui, vocal_effects_gui, background_effects_gui],
            )

        target_format_gui = format_conf()
        button_base = button_conf()
        output_base = output_conf()

        button_base.click(
            sound_separate,
            inputs=[
                aud,
                stem_gui,
                main_gui,
                dereverb_gui,
                vocal_effects_gui,
                background_effects_gui,
                vocal_reverb_room_size_gui, vocal_reverb_damping_gui, vocal_reverb_dryness_gui, vocal_reverb_wet_level_gui,
                vocal_delay_seconds_gui, vocal_delay_mix_gui, vocal_compressor_threshold_db_gui, vocal_compressor_ratio_gui,
                vocal_compressor_attack_ms_gui, vocal_compressor_release_ms_gui, vocal_gain_db_gui,
                background_highpass_freq_gui, background_lowpass_freq_gui, background_reverb_room_size_gui,
                background_reverb_damping_gui, background_reverb_wet_level_gui, background_compressor_threshold_db_gui,
                background_compressor_ratio_gui, background_compressor_attack_ms_gui, background_compressor_release_ms_gui,
                background_gain_db_gui, target_format_gui,
            ],
            outputs=[output_base],
        )

        gr.Examples(
            examples=[
                [
                    "./test.mp3",
                    "vocal",
                    False,
                    False,
                    False,
                    False,
                    0.15, 0.7, 0.8, 0.2,
                    0., 0., -15, 4., 1, 100, 0,
                    120, 11000, 0.5, 0.1, 0.25, -15, 4., 15, 60, 0,
                ],
            ],
            fn=sound_separate,
            inputs=[
                aud,
                stem_gui,
                main_gui,
                dereverb_gui,
                vocal_effects_gui,
                background_effects_gui,
                vocal_reverb_room_size_gui, vocal_reverb_damping_gui, vocal_reverb_dryness_gui, vocal_reverb_wet_level_gui,
                vocal_delay_seconds_gui, vocal_delay_mix_gui, vocal_compressor_threshold_db_gui, vocal_compressor_ratio_gui,
                vocal_compressor_attack_ms_gui, vocal_compressor_release_ms_gui, vocal_gain_db_gui,
                background_highpass_freq_gui, background_lowpass_freq_gui, background_reverb_room_size_gui,
                background_reverb_damping_gui, background_reverb_wet_level_gui, background_compressor_threshold_db_gui,
                background_compressor_ratio_gui, background_compressor_attack_ms_gui, background_compressor_release_ms_gui,
                background_gain_db_gui,
            ],
            outputs=[output_base],
            cache_examples=False,
        )

        gr.Markdown(RESOURCES)

    return app


if __name__ == "__main__":
    for id_model in UVR_MODELS:
        download_manager(
            os.path.join(MDX_DOWNLOAD_LINK, id_model), mdxnet_models_dir
        )

    app = get_gui(theme)
    app.queue(default_concurrency_limit=40)
    app.launch(
        max_threads=40,
        share=IS_COLAB,
        show_error=True,
        quiet=False,
        debug=IS_COLAB,
    )
