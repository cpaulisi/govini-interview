from typing import List
import re
import logging
import torch
import numpy as np
import pandas as pd


class DisparateDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tensor=None):
        self.data = df
        self.tensor = tensor
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        features = self.data.drop("class", axis=1)
        labels = self.data["class"]
        feature = features.iloc[idx, :].to_numpy()
        if self.tensor:
            # convert to tensor
            feature = torch.Tensor(feature)
            labels = torch.Tensor(labels)
            labels = labels.type(torch.LongTensor)
            label = labels[idx]
            return feature, label
        else:
            label = labels.iloc[idx]
        return feature, label    
    @classmethod
    def run_type_compression(
        self, 
        frame: pd.DataFrame=None, 
        column_exclusions: List=[],
        numeric_proportion_threshold: float=0.5
    ) -> pd.DataFrame:
        """Compress types of values into a standard

        Args:
            frame (pd.DataFrame) : data frame to compress types
            column_exclusions (List) : columns to exclude from compression
            numeric_proportion_threshold (float) : proportion of values in mixed type column
                that must pass numeric validation in order to convert that column to either float or int type
        Returns: 
            frame_comp (pd.DataFrame) : compressed type data
        """
        # if no frame manuall
        if frame is None:
            frame = self.data
        # get column types
        col_types = frame.dtypes.to_dict()
        # replace na value with string literal
        frame = frame.fillna("null")
        # set heuristic inline function for testing non-null string values
        non_null = lambda x: ('null' not in x.lower().replace(" ", "")) \
            and ('nan' not in x.lower().replace(" ", ""))\
            and ('notdefined' not in x.lower().replace(" ", ""))\
            and ('na' not in x.lower().replace(" ", ""))
        # convert object types to string values
        for i, data in enumerate(col_types.items()):
            # column is first index of data item
            col = data[0]
            logging.info(f"Pointing to column in position {i}")
            # log column position
            if col not in column_exclusions:
                logging.info(f"Working compression on column {col}")
                frame[col] = frame[col].astype(str)
                # get all values in series that are not null
                non_null_index = frame[col].apply(non_null)
                non_null_s = frame[col][non_null_index]
                # create inline function for testing if data is numeric, only stripping dashes from left 
                isnum = lambda x: (x.lstrip("-").replace(".", "").isnumeric()) and (x.count(".")<=1)
                isnum_s = non_null_s.apply(isnum)
                isnum_sum, isnum_true = isnum_s.value_counts().sum(), isnum_s.value_counts().get(True)
                # get proportion of numeric strings in non-null series
                # get proportion is numeric values are present
                if isnum_true:
                    isnum_prop = float(isnum_true)/float(isnum_sum)
                else: 
                    isnum_prop = 0
                logging.info(f"Numeric proportion for {col}: {isnum_prop}")
                # strip whitespace
                frame[col] = frame[col].str.strip()
                # check proprtion and convert based upon value
                if isnum_prop >= numeric_proportion_threshold:
                    col_numeric = frame[col].apply(isnum)
                    # check for int values among all numeric values
                    int_check = frame[col][col_numeric]\
                        .apply(lambda x: set(x.split(".")[-1]).issubset("0") or x.isdigit())
                    if int_check.all():
                        # set null type
                        frame.loc[~col_numeric,col] = np.nan
                        # set type as int, with float conversion first
                        # to offset np.nan type data
                        frame[col] = frame[col].astype("float")
                        frame[col] = frame[col].astype("Int64")
                        logging.info(f"Converted column {col} into Int64 type.")
                    else: 
                        # set null type
                        frame.loc[~col_numeric,col] = np.nan
                        # set type as float if non-int numeric values exist
                        frame[col] = frame[col].astype("float")
                        logging.info(f"Converted column {col} into float type.")
                # if non null values are not primarily numeric, then convert to lowercase
                # and remove non-alphanumeric chars
                else:
                    # convert all to lowercase
                    frame[col] = frame[col].apply(str.lower)
                    # create regex pattern for keeping only alphanumeric characters
                    frame.loc[non_null_index, col] = frame[col][non_null_index]\
                        .apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', "", x))
                    # create regex pattern for removing redundant whitespace
                    frame.loc[non_null_index, col] = frame[col][non_null_index]\
                        .apply(lambda x: re.sub(" +", " ", x))
                    # convert all null values to nan
                    frame.loc[~non_null_index,col] = np.nan
                    logging.info(f"Converted column {col} into string type.")
            else:
                logging.info(f"Column {col} excluded.") 
        return frame



