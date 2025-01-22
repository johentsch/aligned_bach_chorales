# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: bach
#     language: python
#     name: bach
# ---

# %%
# %load_ext autoreload 
# %autoreload 2
import os
from typing import Dict, Optional
import pandas as pd
from pymeasuremap.base import MeasureMap
from pymeasuremap.compare import Compare

def are_measure_maps_identical(
        preferred_mms: Dict[int, MeasureMap],
        other_mms: Dict[int, MeasureMap],
        ID: bool = False,
        count: bool = True,
        qstamp: bool = True,
        number: bool = True,
        name: bool = False,
        time_signature: bool = True,
        nominal_length: bool = True,
        actual_length: bool = True,
        start_repeat: bool = True,
        end_repeat: bool = True,
        next: bool = True,
):
    """Assumes the two dicts have the same length and identical keys."""
    for (R, preferred), other in zip(preferred_mms.items(), other_mms.values()):
        if preferred is None or other is None:
            print(f"Skipped R. {R}")
            continue
        comparison = Compare(preferred, other)
        if comparison.all_identical(
            ID=ID, count=count, qstamp=qstamp, number=number, name=name, time_signature=time_signature, nominal_length=nominal_length, 
            actual_length=actual_length, start_repeat=start_repeat, end_repeat=end_repeat, next=next):
            print(f"R. {R} OK.")
        else:
            #diagnosis = comparison.diagnose()
            #diagnosis_str = "\n\t".join(str(e) for e in diagnosis)
            print(f"Mismatch for R. {R}")


def load_measure_maps(directory: str, filenames: pd.Series) -> Dict[int, MeasureMap]:
    """Load measure maps by appending each filename from the series to the directory and loading the filepath.
    The dictionary keys correspond to the index of the series.
    """
    result = {}
    for ix, filename in filenames.items():
        try:
            filepath = os.path.join(directory, filename)
            mm = MeasureMap.from_json_file(filepath)
            result[ix] = mm
        except Exception as e:
            print(f"{filepath} failed with\n\t{e!r}")
            result[ix] = None
    return result
    

def remove_extension(filename):
    try:
        return os.path.splitext(filename)[0]
    except Exception:
        return filename

def remove_filename(rel_path: str):
    return os.path.dirname(rel_path)


# %%
alignment = pd.read_csv("../aligned_files.csv", index_col=0)
alignment

# %%
krn_msc_mm_path = os.path.abspath("../data/craigsapp_krn/measuremaps/")
krn_krn_mm_path = os.path.join(krn_msc_mm_path, "kern")
krn_xml_mm_path = os.path.join(krn_msc_mm_path, "musicxml")
krn_mm_filenames = alignment.krn_file.map(remove_extension) + '.mm.json'

# %% [markdown]
# ## Comparing MMs for `.krn` against those for their `.musicxml` and `.msc` conversions

# %%
krn_msc_mms = load_measure_maps(krn_msc_mm_path, krn_mm_filenames)
krn_krn_mms = load_measure_maps(krn_krn_mm_path, krn_mm_filenames)
krn_xml_mms = load_measure_maps(krn_xml_mm_path, krn_mm_filenames)

# %%
are_measure_maps_identical(krn_krn_mms, krn_xml_mms, number=False, end_repeat=False, next=False)

# %% [markdown]
# ## Comparing analysis MMs against all score MMs

# %%
xml_mm_path = os.path.abspath("../data/MarkGotham_xml/measuremaps/")
xml_mxl_mm_filenames = alignment.xml_file.map(remove_extension) + ".mm.json"
xml_folders = alignment.xml_file.map(remove_filename)
xml_msc_mm_filenames = xml_folders + ".mm.json"
analysis_paths = xml_folders.map(lambda folder: os.path.join(folder, "analysis.mm.json"))

# %%
cap_mm_path = os.path.abspath("../data/DCMLab_cap/measuremaps/")
cap_mm_filenames = alignment.cap_file.map(remove_extension) + ".mm.json"

# %%
xml_mxl_mms = load_measure_maps(xml_mm_path, xml_mxl_mm_filenames)
xml_msc_mms = load_measure_maps(xml_mm_path, xml_msc_mm_filenames)
analysis_mms = load_measure_maps(xml_mm_path, analysis_paths)
cap_mms = load_measure_maps(cap_mm_path, cap_mm_filenames)


# %%
def quick_diagnosis(
        preferred_mms: Dict[int, MeasureMap],
        other_mms: Dict[int, MeasureMap],
        ID: bool = False,
        count: bool = True,
        qstamp: bool = True,
        number: bool = True,
        name: bool = False,
        time_signature: bool = True,
        nominal_length: bool = True,
        actual_length: bool = True,
        start_repeat: bool = True,
        end_repeat: bool = True,
        next: bool = True,
        entries_threshold: Optional[int] = None
):
    """Assumes the two dicts have the same length and identical keys."""
    results = {}
    for (R, preferred), other in zip(preferred_mms.items(), other_mms.values()):
        if preferred is None or other is None:
            print(f"Skipped R. {R}")
            continue
        comparison = Compare(preferred, other)
        result = comparison.quick_diagnosis(
            ID=ID, count=count, qstamp=qstamp, number=number, name=name, time_signature=time_signature, nominal_length=nominal_length, 
            actual_length=actual_length, start_repeat=start_repeat, end_repeat=end_repeat, next=next, entries_threshold=entries_threshold)
        results[R] = result
    return pd.Series(results)

def summarize_quick_diagnosis(
        preferred_mms: Dict[int, MeasureMap],
        other_mms: Dict[int, MeasureMap],
        ID: bool = False,
        count: bool = True,
        qstamp: bool = True,
        number: bool = True,
        name: bool = False,
        time_signature: bool = True,
        nominal_length: bool = True,
        actual_length: bool = True,
        start_repeat: bool = True,
        end_repeat: bool = True,
        next: bool = True,
        entries_threshold: Optional[int] = None
):
    
    comparison = quick_diagnosis(
            preferred_mms, other_mms,
            ID=ID, count=count, qstamp=qstamp, number=number, name=name, time_signature=time_signature, nominal_length=nominal_length, 
            actual_length=actual_length, start_repeat=start_repeat, end_repeat=end_repeat, next=next, entries_threshold=entries_threshold)
    return pd.concat([comparison.value_counts(), comparison.value_counts(normalize=True)], axis=1)

summarize_quick_diagnosis(analysis_mms, xml_mxl_mms, entries_threshold=2)

# %%
entries_threshold = 2

all_bach_summaries = {name: summarize_quick_diagnosis(analysis_mms, 
                                                      score_mms, 
                                                      entries_threshold=entries_threshold)
                      for name, score_mms in (
                          ("krn_original", krn_krn_mms),
                          ("krn_musicxml", krn_xml_mms),
                          ("krn_mscz", krn_msc_mms),
                          ("xml_original", xml_mxl_mms),
                          ("xml_mscz", xml_msc_mms),
                          ("cap_mscz", cap_mms),
                      )
                     }

# %%
summaries_df = pd.concat(all_bach_summaries, axis=1).fillna(0)
summaries_df

# %%
summaries_df.sum()

# %%
