from pathlib import Path

import pandas as pd
from genai_tk.utils.config_mngr import global_config
from pydantic import BaseModel


class Mapper(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    col: pd.Series
    dest_class: type[BaseModel]
    dest_field: str


class Table2NodeMapper(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    df: pd.DataFrame
    key_field: str
    mapping: list[Mapper]

    @classmethod
    def from_excel(
        cls,
        file_path: Path | str,
        key_field: str,
        mapping_config: list[tuple[str, type[BaseModel], str]],
        sheet_name: str | int = 0,
    ) -> "Table2NodeMapper":
        """Load Excel file and create WinLossTable instance.

        Args:
            file_path: Path to the Excel file
            key_field: Name of the column to use as key for lookups
            mapping_config: List of (column_name, dest_class, dest_field) tuples
            sheet_name: Sheet name or index to read (default: 0)

        Returns:
            WinLossTable instance with loaded data
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Build mapping list from config
        mapping = [
            Mapper(col=df[col_name], dest_class=dest_class, dest_field=dest_field)
            for col_name, dest_class, dest_field in mapping_config
        ]

        return cls(df=df, key_field=key_field, mapping=mapping)

    def get_mapping(self, key: str | int | float, target: type[BaseModel]) -> BaseModel | None:
        """Read the row matching the key and return the requested object.

        Args:
            key: Value to search for in the key_field column
            target: Type of object to return (e.g., Opportunity, WinLoss)

        Returns:
            Instance of target class with fields populated from dataframe, or None if not found
        """
        # Convert key to match the dataframe column type if needed
        search_key = key
        if isinstance(key, str) and pd.api.types.is_numeric_dtype(self.df[self.key_field]):
            try:
                search_key = float(key) if "." in str(key) else int(key)
            except ValueError:
                pass

        # Find the row where key_field matches the provided key
        mask = self.df[self.key_field] == search_key
        matching_indices = self.df.index[mask]

        if len(matching_indices) == 0:
            return None

        # Get the first matching row position
        row_position = self.df.index.get_loc(matching_indices[0])

        # Build kwargs for the target class from matching mappers
        kwargs = {}
        for mapper in self.mapping:
            if mapper.dest_class == target:
                # Get the value from the mapper's column at the matching position
                raw_value = mapper.col.iloc[row_position]  # type: ignore

                # Convert numpy/pandas types to Python native types
                if pd.isna(raw_value):
                    value = None
                elif hasattr(raw_value, "item"):  # numpy scalar
                    value = raw_value.item()
                else:
                    value = raw_value

                kwargs[mapper.dest_field] = value

        if not kwargs:
            return None

        return target(**kwargs)


if __name__ == "__main__":
    from genai_graph.ekg.baml_client.types import Customer, Opportunity
    from genai_graph.ekg.schema.common_nodes import WinLoss

    TEST_FILE = "misc/Cloud_INFRAdeals_overview_V0.2.xlsx"

    root = global_config().get_dir_path("paths.ekg_data")
    file = root / TEST_FILE
    assert file.exists()

    print("=" * 80)
    print("Example 1: Using from_excel() class method")
    print("=" * 80)

    # Create WinLossTable using the from_excel class method
    t_from_excel = Table2NodeMapper.from_excel(
        file_path=str(file),
        key_field="Atos Opportunity ID",
        mapping_config=[
            ("Account Name", Customer, "name"),
            ("Reason", WinLoss, "result"),
            ("Competition: Lost - what would have made us win?", WinLoss, "reason"),
        ],
        sheet_name=0,
    )

    # Retrieve objects by key
    customer = t_from_excel.get_mapping(key="1442813", target=Customer)
    print("\nCustomer:")
    print(customer)

    winloss = t_from_excel.get_mapping(key="1442813", target=WinLoss)
    print("\nWinLoss:")
    print(winloss)

    print("\n" + "=" * 80)
    print("Example 2: Using constructor directly with more control")
    print("=" * 80)

    # Load the Excel file manually for more control
    df = pd.read_excel(file)
    print(f"\nLoaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()[:5]}...")  # Show first 5 columns

    # Create WinLossTable with custom column transformations
    t = Table2NodeMapper(
        df=df,
        key_field="Atos Opportunity ID",
        mapping=[
            Mapper(col=df["Atos Opportunity ID"].astype(str), dest_class=Opportunity, dest_field="opportunity_id"),
            Mapper(col=df["Opportunity Name"], dest_class=Opportunity, dest_field="name"),
            Mapper(col=df["Account Name"], dest_class=Customer, dest_field="name"),
            Mapper(col=df["Reason"], dest_class=WinLoss, dest_field="result"),
            # Combine two columns into one field
            Mapper(
                col=df["Competition: Lost - what would have made us win?"].fillna("")
                + "\n"
                + df["New Concrete Reasons"].fillna(""),
                dest_class=WinLoss,
                dest_field="reason",
            ),
        ],
    )

    # Get simple objects (no nested dependencies)
    customer = t.get_mapping(key="1442813", target=Customer)
    print("\nCustomer object:")
    print(customer)

    winloss = t.get_mapping(key="1442813", target=WinLoss)
    print("\nWinLoss object:")
    print(winloss)

    # For complex objects with nested dependencies like Opportunity,
    # you need to compose them manually
    print("\nBuilding Opportunity with nested Customer:")
    opp_primitives = {}
    for mapper in t.mapping:
        if mapper.dest_class == Opportunity:
            row_mask = df[t.key_field] == int("1442813")
            row_position = df.index[row_mask].tolist()[0]
            raw_value = mapper.col.iloc[row_position]
            value = raw_value.item() if hasattr(raw_value, "item") else raw_value
            opp_primitives[mapper.dest_field] = value

    if customer and isinstance(customer, Customer):
        opportunity = Opportunity(**opp_primitives, customer=customer)
        print(opportunity)
    else:
        print("Could not retrieve customer")
