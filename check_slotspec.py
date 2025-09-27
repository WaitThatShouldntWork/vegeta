#!/usr/bin/env python3

from vegeta.core.config import Config
from vegeta.utils.database import DatabaseManager

def check_slotspec():
    config = Config()
    db = DatabaseManager(config.database)

    try:
        query = """
        MATCH (ss:SlotSpec)
        RETURN ss.checklist_name AS checklist_name,
               ss.name AS name,
               ss.expect_labels AS expect_labels,
               ss.required AS required
        LIMIT 10
        """
        records = db.execute_query(query)

        print('SlotSpec nodes found:')
        for r in records:
            checklist_name = r.get('checklist_name')
            name = r.get('name')
            expect_labels = r.get('expect_labels')
            required = r.get('required')
            print(f'  {checklist_name}.{name}: {expect_labels} (required: {required})')

        if not records:
            print('  No SlotSpec nodes found!')

    except Exception as e:
        print(f'Error: {e}')
    finally:
        db.close()

if __name__ == '__main__':
    check_slotspec()







