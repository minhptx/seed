from dataclasses import dataclass, field
import jsonlines
import pandas as pd
import json
import collections
import numpy as np
import wikitextparser as wtp

if __name__ == "__main__":
    tables = jsonlines.open("data/wikipedia_history/valid_tables.jsonl")

    key_to_tables = collections.defaultdict(list)

    for table_obj in tables:
        table = pd.DataFrame(table_obj["table"]).astype(str).applymap(lambda x: wtp.remove_markup(x))
        key = (
            "|".join(table.columns.values),
            table_obj["page_id"],
            table_obj["page_title"],
        )
        
        try:
            id = int(table_obj["revision_id"])
        except:
            id = 0
        
        key_to_tables[key].append((id, table))

    data = []

    for key, tables in key_to_tables.items():
        tables.sort(key=lambda x: x[0])
        true_table = tables[-1][1]
        existing_rows = set()
        for revision_id, table in tables[:-1]:
            print("Table", table)
            print("True table", true_table)
            try:
                ground_truth = (table == true_table)
            except ValueError:
                continue
            print(ground_truth)       
            rows = np.where(np.any(~ground_truth, axis=1))[0].tolist()
            print(rows)

            for row in rows:
                example = {
                    "table": table.iloc[row, :].to_json(orient="records"),
                    "label": False,
                    "true_table": true_table.iloc[row, :].to_json(orient="records"),
                    "revision_id": revision_id,
                    "page_title": key[2],
                }

                data.append(example)

                if row not in existing_rows:
                    existing_rows.add(row)
                    example = {
                        "table": true_table.iloc[row, :].to_json(orient="records"),
                        "label": False,
                        "true_table": true_table.iloc[row, :].to_json(orient="records"),
                        "revision_id": revision_id,
                        "page_title": key[2],
                    }
                    data.append(example)

    with jsonlines.open("data/wikipedia_history/data.jsonl", mode="w") as writer:
        writer.write(data)

            
                
