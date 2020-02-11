import os

class Tensorboard:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def log_table(self, tag, table):
        table_str = " | ".join(table[0]) + "\n"
        table_str += "|".join(["---"] * len(table[0])) + "\n"

        for row in table[1:]:
            table_str += " | ".join([str(col) for col in row]) + "\n"

        with open(os.path.join(self.log_dir, tag + ".txt"), "w") as file:
            file.writelines(table_str)

    def log_dict_as_table(self, tag, data):
        if data is not None:
            table = [["name", "value"]]
            for key, value in data.items():
                table.append([key, value])
            self.log_table(tag, table)

    def close(self):
        pass



