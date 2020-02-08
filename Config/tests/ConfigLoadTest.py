from Config import ConfigLoader

def run_test():
    cfg = ConfigLoader.load_config(file_name = "CartPoleConfig.json")
    print("SUCCESSFULLY LOADED FROM CONFIG NAME")
    print(cfg,"\n")
    cfg = ConfigLoader.load_config(file_path = "Configs/ConfigFiles/CartPoleConfig.json")
    print("SUCCESSFULLY LOADED FROM CONFIG FILE PATH")
    print(cfg,"\n")
    print("CONFIG TEST COMPLETE")