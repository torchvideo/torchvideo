from typing import Optional

from .label_set import LabelSet, Label


class CsvLabelSet(LabelSet):
    """LabelSet for a pandas DataFrame or Series. The index of the DataFrame/Series
    is assumed to be the set of video names and the values in a series the label. For a
    dataframe the ``field`` kwarg specifies which field to use as the label

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'video': ['video1', 'video2'],
        ...                    'label': [1, 2]}).set_index('video')
        >>> label_set = CsvLabelSet(df, col='label')
        >>> label_set['video1']
        1

    """

    def __init__(self, df, col: Optional[str] = "label"):
        """

        Args:
            df: pandas DataFrame or Series containing video names/ids and their
                corresponding labels.
            col: The column to read the label from when df is a DataFrame.
        """
        self.df = df
        self._field = col

    def __getitem__(self, video_name: str) -> Label:
        try:
            return self.df[self._field].loc[video_name]
        except KeyError:
            return self.df.loc[video_name]
