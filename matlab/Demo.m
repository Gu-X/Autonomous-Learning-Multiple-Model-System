clear all
clc
close all

load data


Input1.datain=data0; % training input
Input1.dataout=Y0;   % training output
[Output1]=ALMMo(Input1,'Learning');

Input2=Output1;
Input2.datain=data1; % testing input
Input2.dataout=Y1;   % testing output
[Output2]=ALMMo(Input2,'Testing');
Output2.Ye % predicted output 
RMSE=sqrt(mean((Output2.Ye-Y1).^2)) % RMSE of the prediction
