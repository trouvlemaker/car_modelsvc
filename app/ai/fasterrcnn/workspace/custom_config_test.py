from custom_config import config


def main():
    json_path = "configs/dam_cascade_mask_1x.json"
    config.update_from_json(json_path)
    print(config.DATA.TRAIN + config.DATA.VAL)
    

if __name__ == "__main__":
    main()
