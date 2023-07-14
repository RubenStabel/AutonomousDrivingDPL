% Perception
nn(perc_net_version_1_NeSy,[Img],X,[0,1,2,3,4,5,6,7,8,9,10,11]) :: cell_low(Img,X).


% Mapping NN --> X, Y
cell_x(Img, 0) :- cell_low(Img, Z), Z=0.
cell_x(Img, 0) :- cell_low(Img, Z), Z=4.
cell_x(Img, 0) :- cell_low(Img, Z), Z=8.
cell_x(Img, 1) :- cell_low(Img, Z), Z=1.
cell_x(Img, 1) :- cell_low(Img, Z), Z=5.
cell_x(Img, 1) :- cell_low(Img, Z), Z=9.
cell_x(Img, 2) :- cell_low(Img, Z), Z=2.
cell_x(Img, 2) :- cell_low(Img, Z), Z=6.
cell_x(Img, 2) :- cell_low(Img, Z), Z=10.
cell_x(Img, 3) :- cell_low(Img, Z), Z=3.
cell_x(Img, 3) :- cell_low(Img, Z), Z=7.
cell_x(Img, 3) :- cell_low(Img, Z), Z=11.

cell_y(Img, 0) :- cell_low(Img, Z), Z=0.
cell_y(Img, 0) :- cell_low(Img, Z), Z=1.
cell_y(Img, 0) :- cell_low(Img, Z), Z=2.
cell_y(Img, 0) :- cell_low(Img, Z), Z=3.
cell_y(Img, 1) :- cell_low(Img, Z), Z=4.
cell_y(Img, 1) :- cell_low(Img, Z), Z=5.
cell_y(Img, 1) :- cell_low(Img, Z), Z=6.
cell_y(Img, 1) :- cell_low(Img, Z), Z=7.
cell_y(Img, 2) :- cell_low(Img, Z), Z=8.
cell_y(Img, 2) :- cell_low(Img, Z), Z=9.
cell_y(Img, 2) :- cell_low(Img, Z), Z=10.
cell_y(Img, 2) :- cell_low(Img, Z), Z=11.


% Prediction --> 80 pixels
attention_boundary(X,Y) :- X==1, Y=0.


% Prediction ---> 50 pixels
enlarged_boundary(X,Y) :- X==1, Y=1.
enlarged_boundary(X,Y) :- X==2, Y=0.
enlarged_boundary(X,Y) :- X==2, Y=1.


% Prediction ---> ... pixels
out_of_view(X,Y) :- X==0; X==3; Y==2.

% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle

action(Img,1) :- cell_x(Img,X1), cell_y(Img,Y1), attention_boundary(X1,Y1).
action(Img,2)  :-  cell_x(Img,X2), cell_y(Img,Y2), enlarged_boundary(X2,Y2), \+ action(Img,1).
action(Img,0) :-  cell_x(Img,X3), cell_y(Img,Y3), out_of_view(X3,Y3), \+ action(Img,2), \+ action(Img,1).
