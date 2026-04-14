from loguru import logger as eval_logger

def parse_prepare_func(prepare_config):
    eval_logger.info('prepare_config:', prepare_config)

    if prepare_config is None:
        return None
    
    # TO-DO: 根据给定的config，构造出prepare_func并返回
    