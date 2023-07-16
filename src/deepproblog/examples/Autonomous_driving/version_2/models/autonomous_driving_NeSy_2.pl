% Perception
% nn(perc_net_AD_V1,[Img, X,Y],Z,[human,nothing]) :: cell(Img,X,Y,Z).
0.1::cell_x(0); 0.0::cell_x(1); 0.3::cell_x(2); 0.0::cell_x(3);0.1::cell_x(4); 0.1::cell_x(5); 0.1::cell_x(6); 0.1::cell_x(7);0.1::cell_x(8); 0.1::cell_x(9).
0.0::cell_y(0); 0.3::cell_y(1); 0.0::cell_y(2);0.1::cell_y(3); 0.1::cell_y(4); 0.1::cell_y(5);0.1::cell_y(6); 0.1::cell_y(7); 0.1::cell_y(8);0.1::cell_y(9).
0.0::cell_c(red); 0.25::cell_c(green); 0.25::cell_c(blue); 0.5::cell_c(black).



% Perception
%nn(perc_net_version_2_NeSy_x,[Img,Speed],X,[0,1,2,3,4,5,6,7,8,9]) :: cell_x(Img,Speed,X).
%nn(perc_net_version_2_NeSy_y,[Img,Speed],Y,[0,1,2,3,4,5,6,7,8,9]) :: cell_y(Img,Speed,Y).
%nn(perc_net_version_2_NeSy_speed,[Img],C,[red,green,blue,black]) :: cell_c(Img,C).


% Danger zones
danger_zone_1(X,Y,Speed) :- ((X*18/(80+(((Speed+1)**2)/5)))**4 + (Y*30/(170+(((Speed+1)**2)/5)*12))**4 - 1) < 0.
danger_zone_2(X,Y,Speed) :- ((X*18/(70+(((Speed+1)**2)/5)))**4 + (Y*30/(100+(((Speed+1)**2)/5)*12))**4 - 1) < 0.
danger_zone_3(X,Y,Speed) :- ((X*18/(60+(((Speed+1)**2)/5)))**4 + (Y*30/(70 +(((Speed+1)**2)/5)*12))**4 - 1) < 0.


% Speed zone actions
map_c_s(red,2.0).
map_c_s(green,4.0).
map_c_s(blue,6.0).
map_c_s(black,8.0).

speed_zone_brake(C,S) :-
    map_c_s(C,Z),
    S > Z.

speed_zone_follow(C,S) :-
    map_c_s(C,Z),
    Zl is Z - 0.1,
    Zh is Z + 0.1,
    Zl < S,
    S < Zh.


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace


action(Speed,1) :-
    (cell_x(X), cell_y(Y), danger_zone_3(X,Y,Speed));
    (cell_c(C1), speed_zone_brake(C1,Speed)).
action(Speed,2) :-
    (cell_x(X), cell_y(Y), danger_zone_2(X,Y,Speed), \+ danger_zone_3(X,Y,Speed)),
    \+ action(Speed,1).
action(Speed,3) :-
    (cell_x(X), cell_y(Y), danger_zone_1(X,Y,Speed), \+ danger_zone_2(X,Y,Speed), \+ danger_zone_3(X,Y,Speed));
    (cell_c(3), speed_zone_follow(C3,Speed)),
    \+ action(Speed,1),
    \+ action(Speed,2).
action(Speed,0) :-
    \+ action(Speed,1),
    \+ action(Speed,2),
    \+ action(Speed,3).



query(action(8,X)).