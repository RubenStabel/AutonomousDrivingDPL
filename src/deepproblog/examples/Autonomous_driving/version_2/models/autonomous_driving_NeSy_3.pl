% Perception
nn(perc_net_version_2_NeSy_x_rel,[Img,Speed],X,[0,1,2,3,4,5,6,7,8,9]) :: cell_x(Img,Speed,X).
nn(perc_net_version_2_NeSy_y_rel,[Img,Speed],Y,[0,1,2,3,4,5,6,7,8,9]) :: cell_y(Img,Speed,Y).


% Danger zones
danger_zone_1(X,Y,Speed) :- ((X*18/(80+(((Speed+1)**2)/5)))**4 + (Y*30/(170+(((Speed+1)**2)/5)*12))**4 - 1) < 0.
danger_zone_2(X,Y,Speed) :- ((X*18/(70+(((Speed+1)**2)/5)))**4 + (Y*30/(100+(((Speed+1)**2)/5)*12))**4 - 1) < 0.
danger_zone_3(X,Y,Speed) :- ((X*18/(60+(((Speed+1)**2)/5)))**4 + (Y*30/(70 +(((Speed+1)**2)/5)*12))**4 - 1) < 0.


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace

action(Img,Speed,1) :-
    cell_x(Img,Speed,X),
    cell_y(Img,Speed,Y),
    danger_zone_3(X,Y,Speed).
action(Img,Speed,2) :-
    cell_x(Img,Speed,X),
    cell_y(Img,Speed,Y),
    danger_zone_2(X,Y,Speed), \+ danger_zone_3(X,Y,Speed),
    \+ action(Img,Speed,1).
action(Img,Speed,3) :-
    cell_x(Img,Speed,X),
    cell_y(Img,Speed,Y),
    danger_zone_1(X,Y,Speed), \+ danger_zone_2(X,Y,Speed), \+ danger_zone_3(X,Y,Speed),
    \+ action(Img,Speed,1),
    \+ action(Img,Speed,2).
action(Img,Speed,0) :-
    \+ action(Img,Speed,1),
    \+ action(Img,Speed,2),
    \+ action(Img,Speed,3).
