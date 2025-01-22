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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Aligned Bach Chorales
#
# This script can be run as standalone or, if Jupyter with [Jupytext](https://jupytext.readthedocs.io/) extension are installed, opened as a Jupyter notebook.

# %%
import os
import requests
import html
import pandas as pd

from utils import get_dcml_files

cwd = os.path.abspath('')
print(f"Changing the current working directory to {cwd}")
os.chdir(cwd)


# %% [markdown]
# ## Reading the table from bach-corales.com
#
# Compared to the [table on Wikipedia](https://en.wikipedia.org/wiki/List_of_chorale_harmonisations_by_Johann_Sebastian_Bach), 
# the one found on http://www.bach-chorales.com/BachChoraleTable.htm has the advantage that it includes both the titles based 
# on the chorale melody and on the beginning of the lyrics. We will use it as our main metadata table by preparing the `R`
# column into a numerical `Riemenschneider` index and add the pre-alignment filenames of the various datasets.

# %%
def get_table(url_or_html, 
              n=0,
             na_values=["9999", "ZZZZ"],
             attrs = dict(id="sortable")
             ):
    if url_or_html.startswith('http') or url_or_html.startswith('www'):
        r = requests.get(url_or_html, headers=HEADERS)
        html = r.text
    else:
        html = url_or_html
    table = pd.read_html(html, na_values=na_values, attrs=attrs)[n]
    return table
    
# the file was created by copying the HTML source code for the table from http://www.bach-chorales.com/BachChoraleTable.htm
# This detour was taken because pandas was unable to read it entirely directly from the URL
print(f"Reading the table from http://www.bach-chorales.com/BachChoraleTable.htm")
with open("BCT_html_source", "r") as f:
    table_source = f.read()

# bct = get_table("http://www.bach-chorales.com/BachChoraleTable.htm")
bct = get_table(table_source)
bct

# %% [markdown]
# ## Reading in metadata from craigsapp/bach-370-chorales

# %%
print("Reading ../craigsapp_krn/index.hmd")
krn_table = pd.read_csv("../craigsapp_krn/index.hmd", sep='\t', skiprows=2)
krn_table.columns = [c.strip("* ") for c in krn_table.columns]
info_regex = r"<link>(?P<title>.*?)<\/link>(?:, <small>(?:BWV )?(?P<bwv1>\d+)\/?(?P<bwv2>\d+)?.*?<\/small>)? *(?:\((?P<mode>\S+?)\))?"
expanded_description = krn_table.description.map(html.unescape).str.extract(info_regex)
bwv = expanded_description.bwv1 + ('.' + expanded_description.bwv2).fillna('')
krn_metadata = pd.concat([krn_table.iloc[:, :-1], bwv.rename('bwv'), expanded_description], axis=1).iloc[:-1]
krn_metadata.index = krn_metadata.sort.map(int).rename('Riemenschneider')
krn_metadata.dtypes.to_frame('dtypes').to_csv("krn_metadata_dtypes.csv")
krn_metadata.to_csv("krn_metadata.csv", index=False)
krn_metadata

# %% [markdown]
# ## Reading in filenames from DCMLab/bach_chorales

# %%
print("Discovering files in ../DCMLab_cap/MS3")
dcml_chorales = os.path.expanduser("../DCMLab_cap/MS3")
cap_files = get_dcml_files(dcml_chorales, extension='.mscx', remove_extension=False)
title_list = [fname_title[0] if fname_title else None for fname_title in cap_files.values()]

# %% [markdown]
# ## Load the chorales table from Wikipedia
#
# This is the table found on https://en.wikipedia.org/wiki/List_of_chorale_harmonisations_by_Johann_Sebastian_Bach, converted to CSV via https://wikitable2csv.ggor.de/.

# %%
wikitable = pd.read_csv("389_chorale_settings.csv", dtype={
    "389": "Int64",
    "Riemenschneider": "Int64",
})


# %% [markdown]
# ## Assemble the `riemenschneider.csv` and `mapping_table.csv` table
#
# The former will serve the mapping between the 3 datasets (seemingly) organized according to the Riemenschneider catalogue. Whereas the latter will be used to align Riemenschneider chorales with Breitkopf's "purple bible", *389 Choralgesänge*. 

# %%
def clean_bwv(bwv_str):
    result = bwv_str.strip('( )')
    split_result = result.split('.')
    if len(split_result) > 1:
        split_result[-1] = str(int(split_result[-1]))
    result = '.'.join(split_result)
    if result == '145a':
        result = '145-a'
    return result

def make_bwv_column(S):
    S = S.fillna('').str.strip('() ')
    S = S.str.split("=")
    S = S.map(lambda l: [E for elem in l if (E := elem.strip())])
    S = S.map(lambda l: clean_bwv(l[0]) if l else pd.NA)
    return S.rename('bwv')

def make_integer_column(S):
    return S.str.split("[.=]").map(lambda l: int(l[0])).astype("Int64")

print("Assembling metadata...")
riemenschneider = bct[bct.R.notna()].copy()
riemenschneider["Riemenschneider"] = make_integer_column(riemenschneider.R)
riemenschneider = riemenschneider.set_index("Riemenschneider").sort_index()
riemenschneider.insert(3, "CPE", make_integer_column(riemenschneider.B))

mapping_table = pd.merge(
    left=riemenschneider,
    right=wikitable,
    how="outer",
    on="Riemenschneider",
    suffixes=(None, "_wiki")
).sort_values("389").reset_index(drop=True)
mapping_table.to_csv("mapping_table.csv", index=False)
print("Stored mapping_table.csv")

riemenschneider = pd.concat([
    krn_metadata.title.rename('krn_title'),
    pd.Series([f"chor{str(i).zfill(3)}.krn" if i != 150 else None for i in range(1, 372)], index=riemenschneider.index, name='krn_file'),
    pd.Series([f"{str(i).zfill(3)}/short_score.mxl" for i in range(1, 372)], index=riemenschneider.index, name='xml_file'),
    pd.Series(title_list, index=riemenschneider.index, name='cap_file'),
    make_bwv_column(riemenschneider.BWV).rename('bwv'),
    riemenschneider
], axis=1)
riemenschneider.to_csv("riemenschneider.csv")
print("Stored metadata to riemenschneider.csv")
riemenschneider

# %%
mapping_table
