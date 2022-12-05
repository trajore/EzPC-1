from onnx.backend.base import BackendRep

from utils import logger, Party
from utils.backend_helper import decl, comment, take_input, delete_variable, give_output, if_stmnt, iterate_list
from utils.cyborg_func_calls import Operator
from utils.nodes import Node, Input, Output, print_nodes
from utils.onnx_nodes import OnnxNode


def prepare_input(code_list, node, var_dict, input_taken, indent):
    """
    Adds code for Input Nodes in Code-List in CPP Format.
    :param code_list: Code-List in CPP Format.
    :param node: Input Node to be processed.
    :param var_dict: Variable Dictionary.
    :param input_taken: List of variables already input to update it with new inputs.
    :param indent: Space Indentation.
    :return: NA
    """
    if isinstance(node, Input):
        code_list.append(
            comment(
                f"Declaration and Input for variable {node.name} of shape {node.shape} as {var_dict[node.name]}",
                indent + 1,
            )
        )
        # code_list.append(decl(var_dict[node.name], node.data_type, node.shape, indent + 1))
        code_list.append(
            take_input(var_dict[node.name], node.shape, node.party, indent + 1)
        )
        code_list.append("\n\n")
        input_taken.append(node.name)


def prepare_func(code_list, node, var_dict, value_info, input_taken, indent):
    """
    Adds code for Operator Nodes in Code-List in CPP Format.
    :param code_list: Code-List in CPP Format.
    :param node: Input Node to be processed.
    :param var_dict: Variable Dictionary.
    :param value_info: Dictionary {var}->(data-type,shape)
    :param input_taken: List of variables already input to check if arguments still need to be input.
    :param indent: Space Indentation.
    :return: NA
    """
    operator = getattr(Operator, node.op_type)
    code_list.append(
        operator(
            node.attrs, node.inputs, node.outputs, value_info, var_dict, indent + 1
        )
    )


def prepare_output(code_list, node, var_dict, indent):
    """
    Adds code for Input Nodes in Code-List in CPP Format.
    :param code_list: Code-List in CPP Format.
    :param node: Input Node to be processed.
    :param var_dict: Variable Dictionary.
    :param indent: Space Indentation.
    :return: NA
    """
    if isinstance(node, Output):
        code_list.append(
            comment(
                f"Output of variable '{node.name}' of shape {node.shape} as {var_dict[node.name]} to {node.party.name}",
                indent + 1,
            )
        )
        code_list.append(
            give_output(var_dict[node.name], node.shape, node.party, indent + 1)
        )
        code_list.append("\n\n")


def prepare_export(program, var_dict, value_info):
    """
    Prepares the Program List for export by converting it into cpp format.
    :param program: Program List having a list of Input, Nodes and Output nodes classes.
    :param var_dict: Variable Dictionary.
    :param value_info: Dictionary {var}->(data-type,shape).
    :return: Code-List in CPP Format.
    """
    code_list = []
    indent = 1
    input_taken = []  # list of variables already input
    input_dict = dict()
    logger.info("Starting Export...")

    # Check nodes for assertions and modifications
    for node in program:
        func = getattr(OnnxNode, node.op_type)
        func(node)

    # Start CPP program
    code_list.append('#include <iostream>\n#include <vector>\n#include "layers.h"\n#include "softmax.h"\n#include <cmath>\n#include <iomanip>\n\n')
    code_list.append(
    '''int main(int __argc, char **__argv){
        
      // This initializes all backend specific parameters. which can be declared in included file.
      __init(__argc, __argv);\n'''
    )

    # Input
    code_list.append(f"{'   ' * (indent+1)}Tensor4D<mode> input({iterate_list(program[0].shape)});\n")

    code_list.append(f"{'   ' * (indent+1)}auto model = Sequential<u64>({'{'}")
    for node in program:
        if isinstance(node, Node):
            prepare_func(code_list, node, var_dict, value_info, input_taken, indent+1)

    code_list.append(f"{'   ' * (indent+1)}{'}'});\n\n")

    code_list.append(if_stmnt("model.load('input_weights.inp');",indent+1,Party.BOB))

    code_list.append(f"{'   ' * (indent+1)}model.forward(image);\n")

    code_list.append(f"{'   ' * (indent+1)}model.activation.print();\n")

    code_list.append("      return 0;\n")
    code_list.append("}")
    logger.info("Completed Export.")

    return code_list


class CyborgBackendRep(BackendRep):
    """
    This is FzpcBackendRep Class for representing model in a particular backend rather than general onnx.
    Provides functionalities to export the model currently, can be extended to run models directly in future versions.
    """

    def __init__(self, program, value_info, var_dict, path, file_name):
        self.program_AST = program
        self.value_info = value_info
        self.var_dict = var_dict
        self.path = path
        self.file_name = file_name

    def export_model(self):
        """
        Exports the FzpcBackendRep to Secfloat Backend in .cpp format following the crypto protocols.
        :return: NA
        """
        logger.info("Preparing to export Model.")
        code_list = prepare_export(self.program_AST, self.var_dict, self.value_info)
        logger.info(
            f"Secure Model File Saved in Secfloat format as {self.file_name}_secfloat.cpp"
        )

        with open(self.path + f"/{self.file_name}_secfloat.cpp", "w") as fp:
            fp.write("\n".join(code_list))


export_model = CyborgBackendRep.export_model
