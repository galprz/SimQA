# Generated from /home/galprz/simv2/grammer/SimCode.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SimCodeParser import SimCodeParser
else:
    from SimCodeParser import SimCodeParser

# This class defines a complete listener for a parse tree produced by SimCodeParser.
class SimCodeListener(ParseTreeListener):

    # Enter a parse tree produced by SimCodeParser#parse.
    def enterParse(self, ctx:SimCodeParser.ParseContext):
        pass

    # Exit a parse tree produced by SimCodeParser#parse.
    def exitParse(self, ctx:SimCodeParser.ParseContext):
        pass


    # Enter a parse tree produced by SimCodeParser#error.
    def enterError(self, ctx:SimCodeParser.ErrorContext):
        pass

    # Exit a parse tree produced by SimCodeParser#error.
    def exitError(self, ctx:SimCodeParser.ErrorContext):
        pass


    # Enter a parse tree produced by SimCodeParser#simulation_fn.
    def enterSimulation_fn(self, ctx:SimCodeParser.Simulation_fnContext):
        pass

    # Exit a parse tree produced by SimCodeParser#simulation_fn.
    def exitSimulation_fn(self, ctx:SimCodeParser.Simulation_fnContext):
        pass


    # Enter a parse tree produced by SimCodeParser#exp.
    def enterExp(self, ctx:SimCodeParser.ExpContext):
        pass

    # Exit a parse tree produced by SimCodeParser#exp.
    def exitExp(self, ctx:SimCodeParser.ExpContext):
        pass


    # Enter a parse tree produced by SimCodeParser#return_stmt.
    def enterReturn_stmt(self, ctx:SimCodeParser.Return_stmtContext):
        pass

    # Exit a parse tree produced by SimCodeParser#return_stmt.
    def exitReturn_stmt(self, ctx:SimCodeParser.Return_stmtContext):
        pass


    # Enter a parse tree produced by SimCodeParser#repeat_stmt.
    def enterRepeat_stmt(self, ctx:SimCodeParser.Repeat_stmtContext):
        pass

    # Exit a parse tree produced by SimCodeParser#repeat_stmt.
    def exitRepeat_stmt(self, ctx:SimCodeParser.Repeat_stmtContext):
        pass


    # Enter a parse tree produced by SimCodeParser#if_stmt.
    def enterIf_stmt(self, ctx:SimCodeParser.If_stmtContext):
        pass

    # Exit a parse tree produced by SimCodeParser#if_stmt.
    def exitIf_stmt(self, ctx:SimCodeParser.If_stmtContext):
        pass


    # Enter a parse tree produced by SimCodeParser#comperation_operator.
    def enterComperation_operator(self, ctx:SimCodeParser.Comperation_operatorContext):
        pass

    # Exit a parse tree produced by SimCodeParser#comperation_operator.
    def exitComperation_operator(self, ctx:SimCodeParser.Comperation_operatorContext):
        pass


    # Enter a parse tree produced by SimCodeParser#condition_expression.
    def enterCondition_expression(self, ctx:SimCodeParser.Condition_expressionContext):
        pass

    # Exit a parse tree produced by SimCodeParser#condition_expression.
    def exitCondition_expression(self, ctx:SimCodeParser.Condition_expressionContext):
        pass


    # Enter a parse tree produced by SimCodeParser#assignment_stmt.
    def enterAssignment_stmt(self, ctx:SimCodeParser.Assignment_stmtContext):
        pass

    # Exit a parse tree produced by SimCodeParser#assignment_stmt.
    def exitAssignment_stmt(self, ctx:SimCodeParser.Assignment_stmtContext):
        pass


    # Enter a parse tree produced by SimCodeParser#numeric_expression.
    def enterNumeric_expression(self, ctx:SimCodeParser.Numeric_expressionContext):
        pass

    # Exit a parse tree produced by SimCodeParser#numeric_expression.
    def exitNumeric_expression(self, ctx:SimCodeParser.Numeric_expressionContext):
        pass


    # Enter a parse tree produced by SimCodeParser#operator.
    def enterOperator(self, ctx:SimCodeParser.OperatorContext):
        pass

    # Exit a parse tree produced by SimCodeParser#operator.
    def exitOperator(self, ctx:SimCodeParser.OperatorContext):
        pass


    # Enter a parse tree produced by SimCodeParser#open_code_block.
    def enterOpen_code_block(self, ctx:SimCodeParser.Open_code_blockContext):
        pass

    # Exit a parse tree produced by SimCodeParser#open_code_block.
    def exitOpen_code_block(self, ctx:SimCodeParser.Open_code_blockContext):
        pass


    # Enter a parse tree produced by SimCodeParser#close_code_block.
    def enterClose_code_block(self, ctx:SimCodeParser.Close_code_blockContext):
        pass

    # Exit a parse tree produced by SimCodeParser#close_code_block.
    def exitClose_code_block(self, ctx:SimCodeParser.Close_code_blockContext):
        pass


    # Enter a parse tree produced by SimCodeParser#identifier.
    def enterIdentifier(self, ctx:SimCodeParser.IdentifierContext):
        pass

    # Exit a parse tree produced by SimCodeParser#identifier.
    def exitIdentifier(self, ctx:SimCodeParser.IdentifierContext):
        pass


    # Enter a parse tree produced by SimCodeParser#negative_int.
    def enterNegative_int(self, ctx:SimCodeParser.Negative_intContext):
        pass

    # Exit a parse tree produced by SimCodeParser#negative_int.
    def exitNegative_int(self, ctx:SimCodeParser.Negative_intContext):
        pass



del SimCodeParser