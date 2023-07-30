# default python packages
import logging

# our files
from models import baselines as baselines
from models import multitask as multitask
from models import simultaneous_multitask as simmulti


def create_model(labels2idxes, is_multilabel, args):
    """
    Model creation code; create the appropriate model here given kwargs so as to declutter main a little
    @param labels2idxes: a list of lists of label2idx dictionaries for each dataset and each task this model needs
        (e.g., [[{dataset 1 task 1}, {dataset 1 task 2}], [{dataset 2 task 1}, {dataset 2 task 2}]]
    @param is_multilabel: a list of lists of Booleans describing whether each task above is multilabel or not
        (i.e., whether it can have more than one correct label)
    @param args: kwargs generated by main.py
    @return: model (instance from models/), task_setting ("single" or "multi")
    """
    task_setting = "single" if len(labels2idxes) == 1 else "multi"

    # create model
    if args.model == "bert":
        if len(labels2idxes) > 1:
            logging.warning("Creating a singletask model; additional tasks will be ignored.")
            task_setting = "single"

        # labels2idxes[0] better be a list with only one element; the rest are ignored anyway
        # what is sent to the model should be one number and one boolean
        kwargs = {
            "hidden_dropout_prob": args.dropout,
            "num_output_labels": len(labels2idxes[0][0]),
            "is_multilabel": is_multilabel[0][0]
        }

        logging.info("Loading BERT: {bert}".format(bert=args.bert))

        model = baselines.BertSingletask.from_pretrained(args.bert, **kwargs)
    elif args.model == "roberta":
        if len(labels2idxes) > 1:
            logging.warning("Creating a singletask model; additional tasks will be ignored.")
            task_setting = "single"

        # labels2idxes[0] better be a list with only one element; the rest are ignored anyway
        # what is sent to the model should be one number and one boolean
        kwargs = {
            "hidden_dropout_prob": args.dropout,
            "num_output_labels": len(labels2idxes[0][0]),
            "is_multilabel": is_multilabel[0][0]
        }

        logging.info("Loading RoBERTa: {bert}".format(bert=args.bert))

        model = baselines.RobertaSingleTask.from_pretrained(args.bert, **kwargs)

    elif args.model == "multi_alt":
        if len(labels2idxes) == 1:
            logging.warning("Creating a multitask model with only one task. Model will use multitask trappings but "
                            "really be single-task.")

        # everything in labels2idxes better be lists of only one element
        # what is sent to the model should be one list of integers and one list of booleans

        kwargs = {
            "hidden_dropout_prob": args.dropout,
            "num_output_labels": [len(l2i[0]) for l2i in labels2idxes],
            "is_multilabel": [im[0] for im in is_multilabel]
        }

        logging.info("Loading BERT: {bert}".format(bert=args.bert))

        model = multitask.BertMultitask.from_pretrained(args.bert, **kwargs)

    elif args.model == "multi_alt-roberta":
        if len(labels2idxes) == 1:
            logging.warning("Creating a multitask model with only one task. Model will use multitask trappings but "
                            "really be single-task.")

        # everything in labels2idxes better be lists of only one element
        # what is sent to the model should be one list of integers and one list of booleans

        kwargs = {
            "hidden_dropout_prob": args.dropout,
            "num_output_labels": [len(l2i[0]) for l2i in labels2idxes],
            "is_multilabel": [im[0] for im in is_multilabel]
        }

        logging.info("Loading RoBERTa: {bert}".format(bert=args.bert))

        model = multitask.RobertaMultitask.from_pretrained(args.bert, **kwargs)
    elif args.model == "multi":
        # create model
        if len(labels2idxes) > 1:
            raise ValueError("The Multi model is intended for use with only one dataset!")
        if len(labels2idxes[0]) != 2:
            raise ValueError("The Multi model is intended for use with a dataset with two tasks!")

        # we expect labels2idxes to be a list with just one list of two elements, i.e., [[stress, emotion]]
        # what is sent to the model is one list of numbers and one list of booleans
        kwargs = {
            "hidden_dropout_prob": args.dropout,
            "num_output_labels": [len(l2i) for l2i in labels2idxes[0]],
            "is_multilabel": is_multilabel[0]
        }

        logging.info("Loading BERT: {bert}".format(bert=args.bert))

        model = simmulti.SimultaneousMultitask.from_pretrained(args.bert, **kwargs)
    elif args.model == "multi-roberta":
        # create model
        if len(labels2idxes) > 1:
            raise ValueError("The Multi model is intended for use with only one dataset!")
        if len(labels2idxes[0]) != 2:
            raise ValueError("The Multi model is intended for use with a dataset with two tasks!")

        # we expect labels2idxes to be a list with just one list of two elements, i.e., [[stress, emotion]]
        # what is sent to the model is one list of numbers and one list of booleans
        kwargs = {
            "hidden_dropout_prob": args.dropout,
            "num_output_labels": [len(l2i) for l2i in labels2idxes[0]],
            "is_multilabel": is_multilabel[0]
        }

        logging.info("Loading RoBERTa: {bert}".format(bert=args.bert))

        model = simmulti.RobertaSimultaneousMultitask.from_pretrained(args.bert, **kwargs)
    else:
        raise NotImplementedError("Unknown model type: {model}".format(model=args.model))

    return model, task_setting
