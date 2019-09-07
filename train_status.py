class TrainConf:
    # dataset
    weibo = 'weibo'
    msra = 'msra'

    # radical conf
    with_radical = 1
    without_radical = 0

    # network
    conv_gru = 'convgru'
    cnn = 'cnn'
    bilstm = 'bilstm'

    # tagger
    bigru_crf = 'bigrucrf'
    bilstm_crf = 'bilstmcrf'
