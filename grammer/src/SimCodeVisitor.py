# Generated from /home/galprz/simv2/grammer/SimCode.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SimCodeParser import SimCodeParser
else:
    from SimCodeParser import SimCodeParser

# This class defines a complete generic visitor for a parse tree produced by SimCodeParser.

class SimCodeVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SimCodeParser#parse.
    def visitParse(self, ctx:SimCodeParser.ParseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#error.
    def visitError(self, ctx:SimCodeParser.ErrorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#simulation_fn.
    def visitSimulation_fn(self, ctx:SimCodeParser.Simulation_fnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#exp.
    def visitExp(self, ctx:SimCodeParser.ExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#return_stmt.
    def visitReturn_stmt(self, ctx:SimCodeParser.Return_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#repeat_stmt.
    def visitRepeat_stmt(self, ctx:SimCodeParser.Repeat_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#if_stmt.
    def visitIf_stmt(self, ctx:SimCodeParser.If_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#comperation_operator.
    def visitComperation_operator(self, ctx:SimCodeParser.Comperation_operatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#condition_expression.
    def visitCondition_expression(self, ctx:SimCodeParser.Condition_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#assignment_stmt.
    def visitAssignment_stmt(self, ctx:SimCodeParser.Assignment_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#numeric_expression.
    def visitNumeric_expression(self, ctx:SimCodeParser.Numeric_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#operator.
    def visitOperator(self, ctx:SimCodeParser.OperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#open_code_block.
    def visitOpen_code_block(self, ctx:SimCodeParser.Open_code_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#close_code_block.
    def visitClose_code_block(self, ctx:SimCodeParser.Close_code_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#identifier.
    def visitIdentifier(self, ctx:SimCodeParser.IdentifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimCodeParser#negative_int.
    def visitNegative_int(self, ctx:SimCodeParser.Negative_intContext):
        return self.visitChildren(ctx)



del SimCodeParser