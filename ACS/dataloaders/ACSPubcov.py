# primary import
import os

# third party imports
import numpy as np
from folktables import ACSDataSource, BasicProblem, public_coverage_filter

"""
Code adapted from the following work:

@inproceedings{hansen2024multicalibration,
  title={When is Multicalibration Post-Processing Necessary?},
  author={Hansen, Dutch and Devic, Siddartha and Nakkiran, Preetum and Sharan, Vatsal},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
"""

def groups_map(features_df, groups="default"):

    """
    Available features for defining groups:
        'AGEP':     a
        'SCHL':     a
        'MAR':      a
        'SEX':      a
        'DIS':      a
        'ESP':      a
        'CIT':      a
        'MIG':      a
        'MIL':      a
        'ANC':      a
        'NATIVITY': a
        'DEAR':     a
        'DEYE':     a
        'DREM':     a
        'PINCP':    a
        'ESR':      a
        'FER':      a
        'RAC1P':    a
    """

    default_groups = [
        "Black_Adults",
        "Black_Females",
        "Women",
        "Never_Married",
        "American_Indian",
        "Seniors",
        "White_Women",
        "Multiracial",
        "White_Children",
        "Asian",
    ]

    race_groups = {
        "race-{0}".format(i): np.where(features_df["RAC1P"] == i)[0]
        for i in range(1, 10)
    }

    groups_map = {
        "Black_Adults": np.where(
            (features_df["RAC1P"] == 2)
            & (features_df["AGEP"] >= 18)
            & (features_df["AGEP"] <= 99)
        )[0],
        "Black_Females": np.where(
            (features_df["SEX"] == 2) & (features_df["RAC1P"] == 2)
        )[0],
        "Women": np.where((features_df["SEX"] == 2))[0],
        "Never_Married": np.where((features_df["MAR"] == 5))[0],
        "American_Indian": np.where((features_df["RAC1P"] == 3))[0],
        "Seniors": np.where((features_df["AGEP"] >= 65))[0],
        "White_Women": np.where(
            (features_df["SEX"] == 2)
            & (features_df["AGEP"] >= 18)
            & (features_df["RAC1P"] == 1)
        )[0],
        "Multiracial": np.where((features_df["RAC1P"] == 9))[0],
        "White_Children": np.where(
            (features_df["AGEP"] < 18) & (features_df["RAC1P"] == 1)
        )[0],
        "Asian": np.where((features_df["RAC1P"] == 6))[0],
        **race_groups,
    }

    if groups == "default":
        return {gp: groups_map[gp] for gp in default_groups}
    elif groups == "all":
        return groups_map
    elif groups == "alternate":
        return None
    else:
        raise ValueError("Invalid group type")


def load_ACSPubcov_no_sex(states=["CA"], groups="all"):
    """
    ACSIncome dataset, without biological sex features.
    sex-dependent groups added, however.
    """
    return load_ACSPubcov(states, drop_features=["SEX"], groups=groups)


def load_ACSPubcov_no_race(states=["CA"], groups="all"):
    """
    ACSIncome dataset, without race features.
    Race-dependent groups added, however.

    """
    return load_ACSPubcov(states, drop_features=["RAC1P"], groups=groups)


def load_ACSPubcov_no_sex_and_race(states=["CA"], groups="all"):
    """
    ACSIncome dataset, without race features.
    Race-dependent groups added, however.

    """
    return load_ACSPubcov(states, drop_features=["SEX", "RAC1P"], groups=groups)


def load_ACSPubcov(states=["CA"], drop_features=[], groups="default"):
    """
    Dataset provides income data and demographic information
    about US citizens. This data comes from the American Community
    Survey (ACS) Public Use Microdata Sample (PUMS) files, which are managed
    by the US Census Bureau. Our paper studies the 2018 data for California,
    though more data is available.

    Website:
        https://github.com/socialfoundations/folktables

    Original publication:
        @article{ding2021retiring,
            title={Retiring Adult: New Datasets for Fair Machine Learning},
            author={Ding, Frances and Hardt, Moritz and Miller, John and Schmidt, Ludwig},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            year={2021}
        }
    License:
        While Folktables provides API for downloading ACS data, usage of this data
        is governed by the terms of use provided by the Census Bureau.
        For more information, see https://www.census.gov/data/developers/about/terms-of-service.html.
    """
    pubcov_feature_list = [
        "AGEP",
        "SCHL",
        "MAR",
        "SEX",
        "DIS",
        "ESP",
        "CIT",
        "MIG",
        "MIL",
        "ANC",
        "NATIVITY",
        "DEAR",
        "DEYE",
        "DREM",
        "PINCP",
        "ESR",
        "FER",
        "RAC1P",
    ]

    state_str = ""
    for state in states:
        state_str += state + "_"

    DIR = "data/ACS/{0}/".format(state_str[:-1])
    data_source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person", root_dir=DIR
    )

    # check if we need to download
    dl = True
    if os.path.exists(DIR):
        dl = False
    state_data = data_source.get_data(states=states, download=dl)

    # Income parsed by threshold 50000, for convenience
    ACSPubcov_binary = BasicProblem(
        features=pubcov_feature_list,
        target="PUBCOV",
        target_transform=lambda x: x == 1,
        preprocess=public_coverage_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    # filter groups from panda dataframe
    features_df, targets_df, _ = ACSPubcov_binary.df_to_pandas(state_data)
    gm = groups_map(features_df, groups)

    # record in list
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    # drop features
    features_df = features_df.drop(drop_features, axis=1)

    # return data
    X = features_df.values
    y = targets_df.values.reshape(-1)

    return X, y, (gps, gp_names)
