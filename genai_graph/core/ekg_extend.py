from pandas import DataFrame
from pydantic import BaseModel


"""
conn.execute(
    """
    LOAD FROM df
    MERGE (p:Person {name: name})
    MERGE (i:Item {name: item})
    MERGE (p)-[:PURCHASED]->(i)
    ON MATCH SET p.current_city = current_city
    ON CREATE SET p.current_city = current_city
    """
)

"""

def extend_node(type: BaseModel, map_def: BaseModel): ...


def merge_with_df(type: BaseModel, df: DataFrame): ...
