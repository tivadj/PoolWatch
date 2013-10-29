% The class to store any data (workspace) in variable <b>v</b> for a set
% of static functions.
% This is because Matlab is not able to dynamically update class definition
% when some method/property is changed.
%
% Usage example:
% demo1 = LineAligner.create;  % creates global workspace variable to be operated by LineAligner static functions
%                                LineAligner.create method may be implemented by returning instance of TypeErasedClass.
% LineAligner.run(demo1);      % executes static LineAligner.run method on this variable
%
classdef TypeErasedClass < handle
properties
    v;
end
end