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
#     display_name: base
#     language: python
#     name: base
# ---

# %% [markdown]
# # Aligned Bach Chorales
#
# This script can be run as standalone or, if Jupyter with [Jupytext](https://jupytext.readthedocs.io/) extension are installed, opened as a Jupyter notebook.

# %%
import os
from typing import Dict, Optional
import pandas as pd
#pd.set_option('display.max_rows', 500)
import ms3

from utils import get_dcml_files

cwd = os.path.abspath('')
print(f"Changing the current working directory to {cwd}")
os.chdir(cwd)

DATA_FOLDER = os.path.abspath("../data")
assert os.path.isdir(DATA_FOLDER), f"Directory not found: {DATA_FOLDER}"

# %% [markdown]
# ## Loading notes

# %%
def load_notes_tables(number_filepath_tuples):
    result = {}
    for number, filepath in number_filepath_tuples:
        if filepath is None:
            result[number] = None
            continue
        df = ms3.load_tsv(filepath)
        result[number] = df
    return result


# %%
dcml_notes_path = os.path.join(DATA_FOLDER, "DCMLab_cap", "notes")
print(f"Loading notes tables from {dcml_notes_path}...")
number_filepath_dict = {number: os.path.join(dcml_notes_path, basename_title[0]) for number, basename_title in get_dcml_files(dcml_notes_path, extension='.tsv', remove_extension=False).items() if basename_title}
CAP = load_notes_tables(number_filepath_dict.items())


# %% [markdown]
# ## Creating pitch-class vectors

# %%
def get_pcv(df, 
            column: str = 'tpc', 
            n_mcs: Optional[int] = None):
    """Sums up durations for the pitch class in 'column' for the first 'n_mcs' MCs."""
    if n_mcs:
        if df.loc[0, 'mn_onset'] >= 2:
            # has anacrusis
            n_mcs += 1
        selector = df.mc <= n_mcs
        df = df[selector]
    try:
        pcv = df.groupby(column).duration_qb.sum()
    except:
        print(df)
        raise
    return pcv

get_pcv(CAP[1])


# %%
def get_concatenated_pcvs(notes_dict: Dict[str, pd.DataFrame],
                          name: str,
                          column: str = 'tpc',
                          n_mcs: Optional[int] = 2,
                         ) -> pd.DataFrame:
    """Calls get_pcv() on each notes table and concatenates one PCV row per piece."""
    pcvs = {number: get_pcv(df, column=column, n_mcs=n_mcs) for number, df in notes_dict.items()}
    concatenated = pd.concat(pcvs.values(), keys=pcvs.keys())
    result = concatenated.unstack().sort_index().fillna(0.0)
    folder_name = f"{column}_{n_mcs}_pcvs" if n_mcs else f"{column}_pcvs"
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, f"{name}.csv")
    result.to_csv(file_path)
    print(f"Stored pitch-class vectors as {file_path}")
    return result

get_concatenated_pcvs(CAP, 'cap')

# %%
krn_notes_path = os.path.join(DATA_FOLDER, "craigsapp_krn", "notes")
print(f"Loading notes tables from {krn_notes_path}...")
number_filepath_tuples = [(file[4:7], os.path.join(krn_notes_path, file)) for file in os.listdir(krn_notes_path) if file.endswith('.tsv')]
KRN = load_notes_tables(number_filepath_tuples)
get_concatenated_pcvs(KRN, 'krn')

# %%
xml_notes_path = os.path.join(DATA_FOLDER, "MarkGotham_xml", "notes")
print(f"Loading notes tables from {xml_notes_path}...")

number_filepath_tuples = [(int(number), os.path.join(xml_notes_path, file)) for file in os.listdir(xml_notes_path) if file.endswith('.tsv') and (number := file[:3]).isdigit()]
XML = load_notes_tables(number_filepath_tuples)
get_concatenated_pcvs(XML, 'xml')
