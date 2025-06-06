# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import logging
import math
from pathlib import Path

from dora.log import LogProgress
import numpy as np
import musdb
import museval
import torch as th

from .apply import apply_model
from .audio import convert_audio, save_audio
from . import distrib
from .utils import DummyPoolExecutor


logger = logging.getLogger(__name__)

save_tracks = [
    "Al James - Schoolboy Facination",
    "Sambasevam Shanmugam - Kaathaadi",
    "The Easton Ellises - Falcon 69",
    ]


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores


def eval_track(references, estimates, win, hop, compute_sdr=True):
    references = references.transpose(1, 2).double()
    estimates = estimates.transpose(1, 2).double()

    new_scores = new_sdr(references.cpu()[None], estimates.cpu()[None])[0]

    if not compute_sdr:
        return None, new_scores
    else:
        references = references.numpy()
        estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]
        return scores, new_scores


def evaluate(solver, compute_sdr=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """

    args = solver.args

    output_dir = solver.folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = solver.folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # we load tracks from the original musdb set
    if args.test.distorted:
        test_set = musdb.DB(args.dset.musdb_distorted, subsets=["test"], is_wav=True)
    elif args.test.reverb:
        test_set = musdb.DB(args.dset.musdb_reverb, subsets=["test"], is_wav=True)
    elif args.test.shifted_left:
        test_set = musdb.DB(args.dset.musdb_shifted_left, subsets=["test"], is_wav=True)
    elif args.test.shifted_both:
        test_set = musdb.DB(args.dset.musdb_shifted_both, subsets=["test"], is_wav=True)
    elif args.test.shifted_both_025:
        test_set = musdb.DB(args.dset.musdb_shifted_both_025, subsets=["test"], is_wav=True)
    elif args.test.mp3_64kbps_combined:
        test_set = musdb.DB(args.dset.musdb_mp3_64kbps_combined, subsets=["test"], is_wav=True)
    elif args.test.mp3_128kbps_combined:
        test_set = musdb.DB(args.dset.musdb_mp3_128kbps_combined, subsets=["test"], is_wav=True)
    elif args.test.mp3_256kbps_combined:
        test_set = musdb.DB(args.dset.musdb_mp3_256kbps_combined, subsets=["test"], is_wav=True)
    elif args.test.mp3_320kbps_combined:
        test_set = musdb.DB(args.dset.musdb_mp3_320kbps_combined, subsets=["test"], is_wav=True)
    elif args.test.mp3_64kbps_direct:
        test_set = musdb.DB(args.dset.musdb_mp3_64kbps_direct, subsets=["test"], is_wav=True)
    elif args.test.automix:
        test_set = musdb.DB(args.dset.musdb_automix, subsets=["test"], is_wav=True)
    elif args.test.vocal_none:
        test_set = musdb.DB(args.dset.musdb_vocal_none, subsets=["test"], is_wav=True)
    elif args.test.vocal_left:
        test_set = musdb.DB(args.dset.musdb_vocal_left, subsets=["test"], is_wav=True)
    elif args.test.vocal_soft:
        test_set = musdb.DB(args.dset.musdb_vocal_soft, subsets=["test"], is_wav=True)
    elif args.test.vocal_loud:
        test_set = musdb.DB(args.dset.musdb_vocal_loud, subsets=["test"], is_wav=True)
    elif args.test.plus10:
        test_set = musdb.DB(args.dset.musdb_plus10, subsets=["test"], is_wav=True)
    elif args.test.plus6:
        test_set = musdb.DB(args.dset.musdb_plus6, subsets=["test"], is_wav=True)
    elif args.test.plus3:
        test_set = musdb.DB(args.dset.musdb_plus3, subsets=["test"], is_wav=True)
    elif args.test.plus1:
        test_set = musdb.DB(args.dset.musdb_plus1, subsets=["test"], is_wav=True)
    elif args.test.plus15:
        test_set = musdb.DB(args.dset.musdb_plus15, subsets=["test"], is_wav=True)
    elif args.test.plus13:
        test_set = musdb.DB(args.dset.musdb_plus13, subsets=["test"], is_wav=True)
    elif args.test.bandpass_300_8000:
        test_set = musdb.DB(args.dset.musdb_bandpass_300_8000, subsets=["test"], is_wav=True)
    elif args.test.RIR1:
        test_set = musdb.DB(args.dset.musdb_RIR1, subsets=["test"], is_wav=True)
    elif args.test.RIR2:
        test_set = musdb.DB(args.dset.musdb_RIR2, subsets=["test"], is_wav=True)
    elif args.test.speech:
        test_set = musdb.DB(args.dset.musdb_speech, subsets=["test"], is_wav=True)
    elif args.test.mono:
        test_set = musdb.DB(args.dset.musdb_mono, subsets=["test"], is_wav=True)
    elif args.test.custom:
        test_set = musdb.DB(args.dset.custom, subsets=["test"], is_wav=True)
    elif args.test.nonhq is None:
        test_set = musdb.DB(args.dset.musdb, subsets=["test"], is_wav=True)
    else:
        # test_set = musdb.DB(args.test.nonhq, subsets=["test"], is_wav=False)
        test_set = None
    src_rate = args.dset.musdb_samplerate

    eval_device = 'cpu'

    model = solver.model
    win = int(1. * model.samplerate)
    hop = int(1. * model.samplerate)

    indexes = range(distrib.rank, len(test_set), distrib.world_size)
    indexes = LogProgress(logger, indexes, updates=args.misc.num_prints,
                          name='Eval')
    pendings = []

    pool = futures.ProcessPoolExecutor if args.test.workers else DummyPoolExecutor
    with pool(args.test.workers) as pool:
        for index in indexes:
            track = test_set.tracks[index]

            mix = th.from_numpy(track.audio).t().float()
            if mix.dim() == 1:
                mix = mix[None]
            mix = mix.to(solver.device)
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()

            if args.test.input_loudness_db is not None:
                gain = math.pow(10.0, args.test.input_loudness_db / 20.0)
                mix = mix * gain
            
            mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(model, mix[None],
                                    shifts=args.test.shifts, split=args.test.split,
                                    overlap=args.test.overlap)[0]
            estimates = estimates * ref.std() + ref.mean()
            estimates = estimates.to(eval_device)

            references = th.stack(
                [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
            if references.dim() == 2:
                references = references[:, None]
            references = references.to(eval_device)
            references = convert_audio(references, src_rate,
                                       model.samplerate, model.audio_channels)
            if args.test.save:
                # Define the base directory and include the dataset name (replace placeholder)
                if args.test.distorted:
                    dataset_name = "distorted"
                elif args.test.reverb:
                    dataset_name = "reverb"
                elif args.test.shifted_left:
                    dataset_name = "shifted_left"
                elif args.test.shifted_both:
                    dataset_name = "shifted_both"
                elif args.test.shifted_both_025:
                    dataset_name = "shifted_both_0.25"
                elif args.test.mp3_64kbps_combined:
                    dataset_name = "mp3_64kbps_combined"
                elif args.test.mp3_128kbps_combined:
                    dataset_name = "mp3_128kbps_combined"
                elif args.test.mp3_256kbps_combined:
                    dataset_name = "mp3_256kbps_combined"
                elif args.test.mp3_320kbps_combined:
                    dataset_name = "mp3_320kbps_combined"
                elif args.test.mp3_64kbps_direct:
                    dataset_name = "mp3_64kbps_direct"
                elif args.test.automix:
                    dataset_name = "automix"
                elif args.test.vocal_none:
                    dataset_name = "vocal_none"
                elif args.test.vocal_left:
                    dataset_name = "vocal_left"
                elif args.test.vocal_soft:
                    dataset_name = "vocal_soft"
                elif args.test.vocal_loud:
                    dataset_name = "vocal_loud"
                elif args.test.plus10:
                    dataset_name = "loudness_+10"
                elif args.test.plus6:
                    dataset_name = "loudness_+6"
                elif args.test.plus3:
                    dataset_name = "loudness_+3"
                elif args.test.plus1:
                    dataset_name = "loudness_+1"
                elif args.test.plus15:
                    dataset_name = "loudness_+15"
                elif args.test.plus13:
                    dataset_name = "loudness_+13"
                elif args.test.bandpass_300_8000:
                    dataset_name = "bandpass_300_8000"
                elif args.test.RIR1:
                    dataset_name = "RIR1"
                elif args.test.RIR2:
                    dataset_name = "RIR2"
                elif args.test.speech:
                    dataset_name = "speech"
                elif args.test.mono:
                    dataset_name = "mono"
                elif args.test.input_loudness_db:
                    dataset_name = f"loudness_{'+' if args.test.input_loudness_db > 0 else ''}{args.test.input_loudness_db}"
                elif args.test.custom:
                    dataset_name = "custom"
                elif args.test.nonhq is None:
                    dataset_name = "musdb"
                else:
                    dataset_name = "nonhq"
                if track.name in save_tracks:
                    base_save_path = Path(f"/mnt/parscratch/users/aca22cyy/estimates/{dataset_name}")
                    folder = base_save_path / track.name # Use your desired base path + track name
                    folder.mkdir(exist_ok=True, parents=True)
                    for name, estimate in zip(model.sources, estimates):
                        save_audio(estimate.cpu(), folder / (name + ".wav"), model.samplerate)

            pendings.append((track.name, pool.submit(
                eval_track, references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)))

        pendings = LogProgress(logger, pendings, updates=args.misc.num_prints,
                               name='Eval (BSS)')
        tracks = {}
        for track_name, pending in pendings:
            pending = pending.result()
            scores, nsdrs = pending
            tracks[track_name] = {}
            for idx, target in enumerate(model.sources):
                tracks[track_name][target] = {'nsdr': [float(nsdrs[idx])]}
            if scores is not None:
                (sdr, isr, sir, sar) = scores
                for idx, target in enumerate(model.sources):
                    values = {
                        "SDR": sdr[idx].tolist(),
                        "SIR": sir[idx].tolist(),
                        "ISR": isr[idx].tolist(),
                        "SAR": sar[idx].tolist()
                    }
                    tracks[track_name][target].update(values)

        all_tracks = {}
        for src in range(distrib.world_size):
            all_tracks.update(distrib.share(tracks, src))

        per_song_stats = []
        for track_name in sorted(all_tracks.keys()):
            song_dict = {"track": track_name}
            for source in model.sources:
                for metric_name, value in all_tracks[track_name][source].items():
                    # Only include SDR or nsdr stats
                    if metric_name.lower() in ("nsdr"):
                        key = f"{source}_{metric_name}"
                        song_dict[key] = value
            per_song_stats.append(song_dict)

        for song in per_song_stats:
            for key in list(song.keys()):
                value = song[key]
                if key.endswith("_SDR") and isinstance(value, list):
                    # Compute mean SDR, ignoring nan
                    song[key] = float(np.nanmean(value))
                    song[key + "_med"] = float(np.nanmedian(value))

        print("Per-song stats (SDR/nsdr only):")
        print(per_song_stats)

        result = {}
        metric_names = next(iter(all_tracks.values()))[model.sources[0]]
        for metric_name in metric_names:
            avg = 0
            avg_of_medians = 0
            for source in model.sources:
                medians = []
                for track in all_tracks.keys():
                    values = all_tracks[track][source][metric_name]
                    # Only compute median if there is at least one non-nan value
                    arr = np.asarray(values)
                    if np.isnan(arr).all():
                        continue  # skip this track for this source/metric
                    medians.append(np.nanmedian(arr))
                if medians:
                    mean = np.mean(medians)
                    median = np.median(medians)
                else:
                    mean = float('nan')
                    median = float('nan')
                result[metric_name.lower() + "_" + source] = mean
                result[metric_name.lower() + "_med" + "_" + source] = median
                avg += mean / len(model.sources)
                avg_of_medians += median / len(model.sources)
            result[metric_name.lower()] = avg
            result[metric_name.lower() + "_med"] = avg_of_medians
        return result
