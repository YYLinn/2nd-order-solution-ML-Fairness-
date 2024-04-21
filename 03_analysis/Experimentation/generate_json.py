import json

def generate_json(metrics_list, filename):
    super_dict = {}
    for metrics in metrics_list:
        for k, v in metrics.items():
            super_dict.setdefault(k, []).append(v)

    json_object = json.dumps(super_dict, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_object)
