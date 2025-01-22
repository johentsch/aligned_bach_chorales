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
#
# The notebook represents a tool that has evolved in the process of aligning the datasets.

# %%
import os
from typing import TypeVar, Union, Any, Tuple, Optional
import pandas as pd
from IPython.display import display 
pd.set_option('display.max_rows', 500)
pandas_object: TypeVar = Union[pd.DataFrame, pd.Series]

PCV_FOLDER = "tpc_2_pcvs" # which pre-computed pitch-class vectors to use

# %% [markdown]
# ## Loading metadata

# %%
R = pd.read_csv("riemenschneider.csv", index_col=0)
R

# %%
cpe_riem_differences = R.CPE[R.CPE != R.index]
print(f"{len(cpe_riem_differences)} differences between Riemenschneider and original CPE.")
print("Riemenschneider => CPE mapping, apart from the systematic deviation from R. 284 onwards:")
display(cpe_riem_differences.loc[:284])


# %%
def reindex_cpe_with_riemenschneider(pandas: pandas_object):
    """This function aligns a DataFrame or Series whose index reflects the (corrected) CPE numbering with the 
    index of the riemenschneider.csv metadata. Corrected CPE numbering is the one that corresponds
    to the princeps Breitkopf edition by Carl Philipp Emanual Bach & Johann Philipp Kirnberger 
    but with the duplicate attribution of number 283 corrected. In other words, this function assumes
    that in the input DataFrame/Series index 284 corresponds to 283.2 (or 283bis) "Herr Jesu Christ, wahr Mensch und Gott"
    and that all following indices correspond to the original CPE numbering plus one. That way, the
    last index 371 correctly corresponds to CPE number 370 and the function needs to align only the
    first 283 rows.
    """
    global R
    first_part_aligend = R.index.take(R.index.get_indexer(R.CPE.loc[:283]))
    pandas = pandas.reindex(R.index) # to fill up any missing rows
    upper = pandas.loc[first_part_aligend]
    lower = pandas.loc[284:]
    result = pd.concat([upper, lower])
    result.index = R.index
    return result


# %% [markdown]
# ## Loading pitch-class vectors

# %%
def load_pcvs(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df

def iter_pcvs(path):
    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
        name = file[:-4]
        filepath = os.path.join(path, f"{name}.csv")
        df = load_pcvs(filepath)
        yield name, df
        if name in ('cap', 'xml'):
            yield name + "_aligned", reindex_cpe_with_riemenschneider(df)

PCVS = dict(iter_pcvs(PCV_FOLDER))

krn = PCVS['krn']
cap = PCVS['cap']
xml = PCVS['xml']
cap_aligned = PCVS['cap_aligned']
xml_aligned = PCVS['xml_aligned']


# %% [markdown]
# ## Computing summed absolute errors between pitch-class vectors of two datasets

# %%
def is_null_row(df: pd.DataFrame) -> pd.Series:
    return (df.isna() | (df == 0)).all(axis=1)

def absolute_error(A: pandas_object, 
                   B: pandas_object) -> pd.Series:
    """Substract two pitch-class vectors or PCV datasets (or a combination) from each other and sum up the absolute differences."""
    A_is_df, B_is_df = isinstance(A, pd.DataFrame), isinstance(B, pd.DataFrame)
    includes_df = A_is_df + B_is_df
    if includes_df:
        null_row_mask = is_null_row(A) if A_is_df else is_null_row(B)
        if includes_df == 2:
            null_row_mask |= is_null_row(B)
    result = (A - B).abs().sum(axis=1, skipna=True)
    if includes_df and null_row_mask.any():
        result = result.where(~null_row_mask)
    return result.rename('absolute_error')

def fill_up_with_zeros(A: pd.DataFrame, 
                       B: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Makes sure both datasets have the same number of rows and columns by adding missing rows (filled with pd.NA) and columns (filled with 0.0).
    """
    A_x, A_y = A.shape
    B_x, B_y = B.shape
    swapped = False
    if A_x != B_x:
        # A to be the shorter one
        swapped = A_x > B_x
        A, B = (B, A) if swapped else (A, B)
        A = A.reindex(B.index)
    if A_y != B_y:
        # A to be the narrower one
        to_be_swapped = A_y > B_y
        A, B = (B, A) if to_be_swapped and not swapped else (A, B)
        A = A.reindex(columns=B.columns, fill_value=0.0)
        swapped ^= to_be_swapped 
    if swapped:
        return B, A
    return A, B
        
    
    

def compute_errors(A: pd.DataFrame, 
                   B: pd.DataFrame,
                   func = absolute_error
                  ):
    """For each row (piece), substract the PCV of dataset A from the one of dataset B and sum up the absolute errors."""
    A, B = fill_up_with_zeros(A, B)
    return func(A, B)

def get_diverging(A, 
                  B, 
                  func=absolute_error,
                  acceptable_error = 0.0
                 ):
    errors = compute_errors(A, B, func=func)
    selector = errors > acceptable_error
    return errors[selector].copy()

def show_errors(A_name, 
                B_name, 
                func = absolute_error,
                acceptable_error = 0.0
               ):
    A, B = PCVS[A_name], PCVS[B_name]
    errors = compute_errors(A, B, func=func)
    selector = errors > acceptable_error
    threshold = '' if acceptable_error == 0 else f" with an acceptable error of up to {acceptable_error}"
    print(f"When comparing pitch class vectors of datasets {A_name!r} and {B_name!r}{threshold}, {(~selector).sum()} pieces match, {selector.sum()} don't:")
    display(errors[selector])

show_errors("cap", "krn")

# %%
show_errors("cap_aligned", "krn")


# %%
def get_best_matches_for_piece(pcv, pcvs, func=absolute_error):
    errors = func(pcv, pcvs) # casts to the shape of pcvs
    min_val = errors.min()
    return errors[errors == min_val]

get_best_matches_for_piece(cap.loc[87], krn)

# %% [markdown]
# ## Trying to match up

# %%
MD_COLS = dict(
    cap = "cap_file",
    krn = "krn_file",
    xml = "xml_file",
    groundtruth = "krn_title",
)
def get_filenames(dataset):
    global R
    if dataset in MD_COLS:
        return R[MD_COLS[dataset]]
    if dataset == 'cap_aligned':
        return reindex_cpe_with_riemenschneider(R.cap_file)
    # not required anymore since the files have been renamed accordin to Riemenschneider 
    # https://github.com/MarkGotham/Chorale-Corpus/commit/b2cafc917b1aa23c247e453326630132d59fc60a
    # if dataset == 'xml_aligned':
    #     return reindex_cpe_with_riemenschneider(R.xml_file)
    raise ValueError(dataset)
    
def print_title(title, frame_symbol="-", main_title=False):
    frame = frame_symbol * len(title)
    if main_title:
        block = f"\n{frame}\n{title}\n{frame}\n"
    else:
        block = f"\n{title}\n{frame}"
    print(block)

def match_dataset(A_name, 
                  B_name, 
                  func=absolute_error,
                  auto_rematch=False,
                 ):
    """If auto_rematch=False (default), pieces are marked for further scrutiny if they perfectly match other pieces, but not the one with the corresponding ID.
    "Marking" means returning the matched ID(s) as a tuple, rather than an integer, which is what the filter_matches() function below reacts to.
    If auto_rematch=True, the first perfect match will be accepted, i.e. returned as integer, even it it has a different ID.
    """
    A, B = fill_up_with_zeros(PCVS[A_name], PCVS[B_name])
    A_filenames = get_filenames(A_name)
    B_filenames = get_filenames(B_name)
    errors = compute_errors(A, B, func=func)
    match_results = []
    print_title(f"Comparing datasets {A_name!r} and {B_name!r}", frame_symbol="=", main_title=True)
    for error, (i, pcv) in zip(errors, A.iterrows()):
        if pcv.isna().all():
            match_results.append([pd.NA, pd.NA, pd.NA])
            continue
        file_to_be_matched = A_filenames.loc[i]
        if error == 0:
            match_results.append([file_to_be_matched, error, i])
            continue            
        matches = get_best_matches_for_piece(pcv, B)
        matches = pd.concat([matches, B_filenames.loc[matches.index]], axis=1)
        print_title(f"{i}: {file_to_be_matched}")
        matched_ids = tuple(matches.index)
        n_matches = len(matched_ids)
        if n_matches == 0:
            print("ERROR, NO MATCHES FOR")
            display(pcv)
            raise ValueError(str(pcv))
        match_no, match_error, filename = next(matches.itertuples())
        if match_error == 0 and auto_rematch:
            match_results.append([file_to_be_matched, error, match_no])
            choices = '' if len(matched_ids) == 1 else f"(selecting the first out of {matched_ids})"
            print(f"{A_name} {i} was automatically matched with {B_name} {match_no} {choices}.")
            continue
        if n_matches == 1:
            if match_error == 0:
                print(f"{A_name} {i} == {B_name} {match_no}")
                print(f"{A_filenames.loc[i]} == {filename}")
            else:
                print(f"{A_name} {i} has most resemblance with {B_name} {match_no} => absolute difference = {match_error}")
                print(f"{A_filenames.loc[i]} ~ {filename}")
        else:
            display(matches)
            match_error = matches.absolute_error.iloc[0]
        match_results.append([file_to_be_matched, match_error, matched_ids])
    return pd.DataFrame(match_results, columns=["file_to_be_matched", "error", "ids"], index=A.index)
        
        
cap_aligned_krn = match_dataset('cap_aligned', 'krn')


# %% [markdown]
# ### DCML (aligned) <-> krn

# %%
def filter_matches(df, unmatched=True):
    """If unmatched=True (default), get those pieces that have not been unequivocally matched and show match candidate(s). Otherwise show those that have been matched."""
    df = df[~df.isna().any(axis=1)]
    mask = pd.to_numeric(df.ids, errors='coerce')
    if unmatched:
        return df[mask.isna()].copy()
    return df[mask.notna()].copy()

filter_matches(cap_aligned_krn)


# %%
def get_unequivocal_matches(df, only_differing=True, acceptable_error = 0.0):
    if only_differing:
        df = filter_matches(df, unmatched=True)
    return df[df.error <= acceptable_error].copy()

unequivocal_matches = get_unequivocal_matches(cap_aligned_krn, only_differing=False)
unequivocal_matches


# %%
def get_tentative_matches(df, acceptable_error = 0.0):
    return df[df.error > acceptable_error].copy()

get_tentative_matches(cap_aligned_krn)

# %%
groundtruth_update = {
    17: ('krn',),  # dcml Alt m. 1, b. 3 has C, not C#
    43: ('cap',), # krn missing E2 in Bass m. 1, b. 1
    51: ('krn',),  # dcml missing Bass transition D3 in m. 1, b. 4.5
    55: ('cap',), # krn writes first two phrases with repeat sign
    57: ('krn',),  # dcml Alto m. 1, b. 2 has Eb not D#
    71: ('cap',), # krn Soprano m. 1, b. 1 has wrong transition via A4
    105: ('krn',), # dcml Bass m. 1, b. 2 has F not E#
    132: ('cap',), # krn processing error reported here: https://github.com/craigsapp/bach-370-chorales/issues/5
    134: ('krn',),  # dcml has three times Ab not G#
    139: ('krn',),  # dcml Bass m.2, b. 2 has G# not Ab
    145: ('krn',),  # dcml Bass m. 1, b. 1 has G, not G#
    150: ('cap',), # krn omits because 5-voice
    174: ('krn',),  # dcml Sopran m. 1, b. 2 has G not F, Bass m. 2 b. 3 has Ab not G#
    194: ('krn',),  # dcml Alto mm. 1-2 has F not E#
    199: ('cap',), # krn Alto m. 0 upbeat has Eb not E
    # the following mismatch was addressed in the krn dataset via commit 4448486 (https://github.com/craigsapp/bach-370-chorales/commit/44484866646498bd521c0d4bac73e72d6ed5f0a3)
    # 204: ('cap',), # krn Alto has additional transition C4 m. 1, b. 4
    216: ('krn',),  # dcml Bass m. 2 has C not B#
    231: ('krn',),  # dcml Bass m. 1, b. 2 has F not F#
    248: ('krn',),  # dcml beginning is set slightly differently
    253: ('krn',),  # dcml Tenor m. 0 has Db not C#
    254: ('krn',),  # dcml Alto m. 2, b. 3f. going down to B
    261: ('krn',),  # dcml has Bb not A#
    270: ('krn',),  # dcml Tenor, first two beats set slightly differently
    276: ('cap',), # krn Tenor m. 1, b. 2 set slightly differently
    283: ('krn',),  # dcml Tenor m. 1, b. 3 has a quarter not an eighth
    284: ('krn',),  # dcml Alto's first two notes C not A
    287: ('cap',), # krn Tenor m. 1, b. 3f set slightly differently
    289: ('cap',), # krn Bass m. 1 hass eighth movement on b. 1 not 2
    292: ('krn',),  # dcml set slightly differently
    293: ('krn',),  # dcml set slightly differently (beginning closer to R. 65)
    302: ('cap',), # same as R. 199: krn Alto m. 0 upbeat has Eb not E
    328: ('cap',), # krn Alto m. 2, b. 2 has D not G
    339: ('krn',),  # dcml Tenor m. 1 b. 4 has Eb not D#
    347: ('krn',),  # dcml has different setting, closer to R. 293
    360: ('cap',), # krn Bass m. 1, b. 1 has ornamenting sixteenths
    371: ('krn',),  # dcml has Bb not A#
}
groundtruth_update.update({cap_missing: ('krn',) for cap_missing in cap_aligned_krn.index[cap_aligned_krn.error.isna()]})


# %%
def show_pcvs(A_name_no_tuple, B_name_no_tuple):
    A_name, A_no = A_name_no_tuple
    B_name, B_no = B_name_no_tuple
    A = PCVS[A_name].loc[A_no].rename(f"{A_name}_{A_no}")
    B = PCVS[B_name].loc[B_no].rename(f"{B_name}_{B_no}")
    error = (A - B).abs().rename("difference")
    return pd.concat([A, B, error], axis=1).T

show_pcvs(("cap_aligned", 293), ("krn", 293))

# %%
groundtruth_index = pd.Series([('krn', 'cap')] * len(unequivocal_matches), index=unequivocal_matches.index).reindex(R.index)
to_be_updated = groundtruth_index.loc[groundtruth_update.keys()]
assert to_be_updated.isna().all(), str(to_be_updated[to_be_updated.notna()])

# %%
groundtruth_index.loc[groundtruth_update.keys()] = list(groundtruth_update.values())
assert groundtruth_index.isna().sum() == 0, f"The groundtruth index has not been filled completely. Missing: {groundtruth_index.index[groundtruth_index.isna()].to_list()}"

# %%
groundtruth_selector = groundtruth_index.map(lambda t: t[0])
krn_selector = groundtruth_selector.index[groundtruth_selector == 'krn']
cap_selector = groundtruth_selector.index[groundtruth_selector == 'cap']
krn_groundtruth = krn.loc[krn_selector]
cap_groundtruth = cap_aligned.loc[cap_selector]
groundtruth_pcvs = pd.concat([krn_groundtruth, cap_groundtruth]).sort_index()
PCVS["groundtruth"] = groundtruth_pcvs
groundtruth_pcvs.to_csv('groundtruth_pcvs.csv')
pd.concat([groundtruth_selector.rename('source_dataset'), groundtruth_pcvs], axis=1).head()

# %%
aligned = pd.concat([
    R.krn_file,
    reindex_cpe_with_riemenschneider(R.cap_file),
], axis=1)
aligned

# %% [markdown]
# ## XML vs. groundtruth

# %%
xml_aligned_krn = match_dataset('xml', 'groundtruth')

# %%
unequivocal_matches = get_unequivocal_matches(xml_aligned_krn, only_differing=False)
unequivocal_matches

# %%
get_tentative_matches(xml_aligned_krn)
# 55 ("057") also writes the beginning as repeats, like the krn scores
# 121 begins with split bar m. 1 and uses first/second ending, like print (although non-sensical)
# 253 uses split bars and first/second ending, like print (although non-sensical)
# 272 same

# %%
aligned = pd.concat([aligned, R.xml_file], axis=1)
aligned.to_csv("../aligned_files.csv")

# %%
