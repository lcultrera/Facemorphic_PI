import datetime

def namer(parameters):
    # Sort the dictionary by keys to ensure consistent order
    now = datetime.datetime.now()
    time = now.strftime("%y-%m-%d_%H-%M-%S")
    #sorted_params = sorted(parameters.items())
    #get time
    # Create a list of key-value pairs as strings
    param_strings = [f"{v}" for k, v in parameters.items()]
    # Join the parameter strings with underscores
    experiment_name = str(time)+"_".join(param_strings)
    return experiment_name