clear all
clc
close all

%% Loading Autompg data
% load autompg_train;
% load autompg_test;
% TRAIN=autompg_train;
% TEST=autompg_test;
% Y0 = TRAIN(:,7);
% data0 = TRAIN(:,1:6);
% Y1 = TEST(:,7);
% data1 = TEST(:,1:6);
%% Loading Delta Ailerons data
% load deltaailerons_train;
% load deltaailerons_test;
% TRAIN=deltaailerons_train;
% TEST=deltaailerons_test;
% Y0 = TRAIN(:,6);
% data0 = TRAIN(:,1:5);
% Y1 = TEST(:,6);
% data1 = TEST(:,1:5);
%% Loading Triazines data
% load triazines_train;
% load triazines_test;
% TRAIN=triazines_train;
% TEST=triazines_test;
% Y0 = TRAIN(:,61);
% data0 = TRAIN(:,1:60);
% Y1 = TEST(:,61);
% data1 = TEST(:,1:60);
%% Loading Autos data
load autos_train;
load autos_test;
TRAIN=autos_train;
TEST=autos_test;
Y0= TRAIN(:,16);
data0 = TRAIN(:,1:15);
Y1 = TEST(:,16);
data1 = TEST(:,1:15);
%% Loading Calhousing data
% load calhousing_train;
% load calhousing_test;
% TRAIN=calhousing_train;
% TEST=calhousing_test;
% Y0 = TRAIN(:,9);
% data0 = TRAIN(:,1:8);
% Y1 = TEST(:,9);
% data1 = TEST(:,1:8);


Input1.datain=data0; % training input
Input1.dataout=Y0;   % training output
[Output1]=ALMMo(Input1,'Learning');

Input2=Output1;
Input2.datain=data1; % testing input
Input2.dataout=Y1;   % testing output
[Output2]=ALMMo(Input2,'Testing');
Output2.Ye % predicted output 
RMSE=sqrt(mean((Output2.Ye-Y1).^2)) % RMSE of the prediction
