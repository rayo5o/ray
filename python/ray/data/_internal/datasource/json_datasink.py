from typing import Any, Callable, Dict, Optional

import pandas as pd
import pyarrow

from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import _resolve_kwargs
from ray.data.datasource.file_datasink import BlockBasedFileDatasink


class JSONDatasink(BlockBasedFileDatasink):
    def __init__(
        self,
        path: str,
        *,
        pandas_json_args_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        pandas_json_args: Optional[Dict[str, Any]] = None,
        file_format: str = "json",
        **file_datasink_kwargs,
    ):
        super().__init__(path, file_format=file_format, **file_datasink_kwargs)

        if pandas_json_args_fn is None:
            pandas_json_args_fn = lambda: {}  # noqa: E731

        if pandas_json_args is None:
            pandas_json_args = {}

        self.pandas_json_args_fn = pandas_json_args_fn
        self.pandas_json_args = pandas_json_args

    def write_block_to_file(self, block: BlockAccessor, file: "pyarrow.NativeFile"):
        writer_args = _resolve_kwargs(self.pandas_json_args_fn, **self.pandas_json_args)
        orient = writer_args.pop("orient", "records")
        lines = writer_args.pop("lines", True)

        df = block.to_pandas()
        # Same default as to_json(): ISO for orient="table", epoch otherwise.
        date_format = writer_args.get("date_format") or (
            "iso" if orient == "table" else "epoch"
        )
        if date_format == "epoch":
            df = _cast_arrow_dates_to_epoch(df, writer_args.get("date_unit", "ms"))

        df.to_json(file, orient=orient, lines=lines, **writer_args)


def _cast_arrow_dates_to_epoch(df: pd.DataFrame, date_unit: str) -> pd.DataFrame:
    """Cast ``pd.ArrowDtype`` date32 and date64 columns to integers in ``date_unit``.

    pandas 3 ``to_json`` raises ``NotImplementedError`` on these columns when it
    writes epoch values. The integers are what pandas < 3 writes for them.
    """
    # Invalid date_unit values are left for to_json() to reject.
    if date_unit not in ("s", "ms", "us", "ns"):
        return df
    # Go by position because column names can repeat.
    positions = [
        i
        for i, dtype in enumerate(df.dtypes)
        if isinstance(dtype, pd.ArrowDtype)
        and pyarrow.types.is_date(dtype.pyarrow_dtype)
    ]
    if not positions:
        return df

    df = df.copy(deep=False)
    for i in positions:
        values = (
            pyarrow.array(df.iloc[:, i])
            .cast(pyarrow.timestamp(date_unit))
            .cast(pyarrow.int64())
        )
        df.isetitem(
            i, pd.Series(values, index=df.index, dtype=pd.ArrowDtype(pyarrow.int64()))
        )
    return df
