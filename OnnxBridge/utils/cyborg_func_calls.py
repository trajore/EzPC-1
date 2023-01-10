import math
from utils import logger, VariableGen
from utils.backend_helper import (
    iterate_list,
    iterate_dict,
    decl,
    comment,
    generate_reshape_vars,
    decl_multiple_int,
    nested_for_reshape_loop,
    iterate_concat_list,
    concat_list,
)


def get_padding(attributes, inputs, output, value_info, var_dict):
    if "auto_pad" in attributes.keys():
        if (
            str(attributes["auto_pad"], "UTF-8") == "NOTSET"
            or str(attributes["auto_pad"], "UTF-8") == "VALID"
        ):
            return attributes["pads"] if "pads" in attributes.keys() else [0, 0, 0, 0]
        else:
            stride_h = attributes["strides"][0]
            stride_w = attributes["strides"][1]
            out_h = value_info[output[0]][1][2]
            out_w = value_info[output[0]][1][3]
            in_h = value_info[inputs[0]][1][2]
            in_w = value_info[inputs[0]][1][3]
            ker_h = (
                value_info[inputs[1]][1][2]
                if "kernel_shape" not in attributes.keys()
                else attributes["kernel_shape"][0]
            )
            ker_w = (
                value_info[inputs[1]][1][3]
                if "kernel_shape" not in attributes.keys()
                else attributes["kernel_shape"][0]
            )
            pad_h = math.ceil(((out_h - 1) * stride_h + ker_h - in_h) / 2)
            pad_w = math.ceil(((out_w - 1) * stride_w + ker_w - in_w) / 2)
            pads = [pad_h, pad_w, pad_h, pad_w]
            return pads
    else:
        return attributes["pads"]
    pass


class Operator:
    """
    Class preparing the Function Calls specific for each function.
    """

    @classmethod
    def Relu(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Relu function call.")
        return str(f"{'   ' * indent}new ReLU<{mode}>(),")

    @classmethod
    def Truncate(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Truncate function call.")
        return str(f"{'   ' * indent}new Truncate<{mode}>(scale),")

    @classmethod
    def Softmax(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Softmax function call.")
        # todo: check format

    @classmethod
    def Conv(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Conv function call.")
        pads = get_padding(attributes, inputs, outputs, value_info, var_dict)

        spatial_size = len(value_info[inputs[0]][1]) - 2
        if spatial_size == 2:
            assert (
                len(inputs) == 2 or len(inputs) == 3
            )  # todo: bias is always there or not
            assert len(attributes["strides"]) == 2
            assert value_info[inputs[1]][1][2:] == tuple(attributes["kernel_shape"])
            CI = value_info[inputs[0]][1][1]
            CO = value_info[outputs[0]][1][1]
            filterShape = value_info[inputs[1]][1][2]
            pad = pads[0]
            stride = attributes["strides"][0]
            isBias = ", true" if len(inputs) == 3 else ""
            return str(
                f"{'   ' * indent}new Conv2D<{mode}, scale>("
                f"{CI}, {CO}, {filterShape}, {pad}, {stride}{isBias}"
                f"),\n"
            ) + cls.Truncate(
                attributes, inputs, outputs, value_info, var_dict, mode, indent
            )

    @classmethod
    def MaxPool(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside MaxPool function call.")
        pads = get_padding(attributes, inputs, outputs, value_info, var_dict)
        filter_shape = attributes["kernel_shape"][0]
        pad = pads[0]
        stride = attributes["strides"][0]
        return str(
            f"{'   ' * indent}new MaxPool2D<{mode}>("
            f"{filter_shape}, {pad}, {stride}"
            f"),"
        )

    @classmethod
    def AveragePool(
        cls, attributes, inputs, outputs, value_info, var_dict, mode, indent
    ):
        logger.debug("Inside AveragePool function call.")
        pads = get_padding(attributes, inputs, outputs, value_info, var_dict)
        filter_shape = attributes["kernel_shape"][0]
        pad = pads[0]
        stride = attributes["strides"][0]
        return str(
            f"{'   ' * indent}new AvgPool2D<{mode}>("
            f"{filter_shape}, {pad}, {stride}"
            f"),"
        )

    @classmethod
    def Flatten(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Flatten function call.")
        return str(f"{'   ' * indent}new Flatten<{mode}>(),")

    @classmethod
    def Reshape(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Reshape function call.")
        # todo : check format

    @classmethod
    def Gemm(cls, attributes, inputs, outputs, value_info, var_dict, mode, indent):
        logger.debug("Inside Gemm function call.")
        inn = value_info[inputs[0]][1][1]
        out = value_info[outputs[0]][1][1]
        isBias = ", true" if len(inputs) == 3 else ""

        return str(
            f"{'   ' * indent}new FC<{mode}, scale>(" f"{inn}, {out}{isBias}" f"),\n"
        ) + cls.Truncate(
            attributes, inputs, outputs, value_info, var_dict, mode, indent
        )
