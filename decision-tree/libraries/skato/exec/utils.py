# -*- coding: utf-8 -*-


def add_argument_from_config(parser, args_filepath):
    import configparser
    config = configparser.ConfigParser()
    config.read(args_filepath)
    for arg in config:
        if arg == "DEFAULT":
            continue

        settings = {}
        for k, v in config[arg].items():
            if k == "type":
                if v == "bool":
                    raise Exception(
                        "Plese use action:store_true/action:store_false instead of type:bool")
                elif v == "int":
                    settings[k] = int
                elif v == "str":
                    settings[k] = str
                elif v == "float":
                    settings[k] = float
            else:
                settings[k] = v
        if "required" in settings:
            settings["required"] = eval(settings["required"])
        if "default" in settings:
            if settings["default"] == "None":
                settings["default"] = None
            elif settings["default"] == "True":
                settings["default"] = True
            elif settings["default"] == "False":
                settings["default"] = False
            else:
                settings["default"] = settings["type"](settings["default"])
        parser.add_argument(arg, **settings)


def main(filepath, global_dict):
    import os
    filename = os.path.basename(filepath)

    import argparse
    parser = argparse.ArgumentParser()

    args_filepath = filepath + ".args"
    if os.path.isfile(args_filepath):
        add_argument_from_config(parser, args_filepath)

    parser.add_argument("--log-level",
                        type=str,
                        default="DEBUG",
                        help="Set logging level (default: 'DEBUG')")
    parser.add_argument("--log-mode",
                        type=str,
                        default="stdout",
                        help="Set logging mode (default: 'stdout')")
    parser.add_argument("--log-dir",
                        type=str,
                        default=".",
                        help="Set path to log directory (default: '.')")
    args = parser.parse_args()

    import logging
    LOG_LEVEL = getattr(logging, args.log_level)
    LOG_FORMAT = "%(asctime)s\t%(levelname)s\t%(message)s"
    if args.log_mode == "stdout":
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    else:
        import sys
        import datetime
        now = datetime.datetime.now()
        log_filename = "%s.%s.log" % (os.path.basename(filename),
                                      now.strftime("%Y%m%d%H%M%S"))
        log_path = os.path.join(args.log_dir, log_filename)
        print(log_path)
        sys.stdout.flush()
        logging.basicConfig(
            handlers=[logging.FileHandler(log_path, args.log_mode, 'utf-8')],
            level=LOG_LEVEL,
            format=LOG_FORMAT)
        # logging.basicConfig(filename=log_path, filemode=args.log_mode,
        #                     level=LOG_LEVEL, format=LOG_FORMAT)

    # e.g. predict_ans.py => predict_ans, predict.ans.py => predict
    main_func_name = filename.split(".")[0]
    try:
        main_func = global_dict[main_func_name]
    except Exception as e:
        raise Exception("%s is not defined" % main_func_name)

    logging.info("Starting %s function in %s" % (main_func_name, filepath))
    main_func(**vars(args))
    logging.info("Finished %s function in %s" % (main_func_name, filepath))
