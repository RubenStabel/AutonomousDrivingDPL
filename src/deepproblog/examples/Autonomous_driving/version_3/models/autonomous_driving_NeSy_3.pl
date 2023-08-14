% Perception
nn(perc_net_version_3_NeSy_danger_pedestrian,[Img,Speed],X,[0,1,2,3]) :: cell_danger_pedestrian(Img,Speed,X).
nn(perc_net_version_3_NeSy_speed_zone,[MNIST],SZ,[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]) :: cell_speed_zone(MNIST,SZ).
nn(perc_net_version_3_NeSy_traffic_light,[Img],X,[green, orange, red]) :: cell_traffic_light(Img,X).

%%%%    TRAFFIC RULES    %%%%
% Rules - pedestrian
ped_brake(Img,Speed) :-
    cell_danger_pedestrian(Img,Speed,D),
    D = 3.
ped_idle(Img,Speed) :-
    cell_danger_pedestrian(Img,Speed,D),
    D = 2.
ped_follow(Img,Speed) :-
    cell_danger_pedestrian(Img,Speed,D),
    D = 1.
ped_acc(Img,Speed) :-
    cell_danger_pedestrian(Img,Speed,D),
    D = 0.

% Rules - speed zone
speed_zone_brake(MNIST,Speed) :-
    cell_speed_zone(MNIST,SZ),
    Speed > SZ.

speed_zone_follow(MNIST,Speed) :-
    cell_speed_zone(MNIST,SZ),
    Sl is SZ - 0.11,
    Speed =< SZ,
    Speed > Sl.

speed_zone_accelerate(MNIST,Speed) :-
    cell_speed_zone(MNIST,SZ),
    Sl is SZ - 0.11,
    Speed =< Sl.


% Rules - distance
brake_dist(Y,Speed) :-
    Acc = 0.1,
    Frames_brake = Speed/(2*Acc),
    Y is (-(2*Acc)*Frames_brake**2)/2 + Speed*Frames_brake.

idle_dist(Y,Speed) :-
    Acc = 0.1,
    Frames_idle = Speed/(Acc/2),
    Y is (-(Acc/2) * Frames_idle ** 2) / 2 + Speed * Frames_idle.


% Rules - traffic light
traffic_light_brake(Img,MNIST,Speed,Tl_y,P_y) :-
    \+ Tl_y = -1,
    Margin = 15,
    Yr is P_y - Tl_y,
    Yr >= 0,
    brake_dist(Yb,Speed),
    Yb_margin is Yb + Margin,
    Yr < Yb_margin,
    Mb is Margin - 1,
    (Speed > 0.2 ; Yr < Mb),
    cell_traffic_light(Img,C),
    (C = red ; C = orange).


traffic_light_idle(Img,MNIST,Speed,Tl_y,P_y) :-
    \+ Tl_y = -1,
    Margin = 15,
    Yr is P_y - Tl_y,
    Yr >= 0,
    Yr < 300,
    ((brake_dist(Yb,Speed),
    Yb_margin is Yb + Margin,
    Yr < Yb_margin,
    Mb is Margin - 1,
    (Speed =< 0.2 ; Yr >= Mb),
    cell_traffic_light(Img,C),
    (C = red ; C = orange))
    ;
    (idle_dist(Yi,Speed),
    Yi_margin is Yi + Margin,
    Yr < Yi_margin,
    cell_traffic_light(Img,C),
    C = orange)).


traffic_light_accelerate(Img,MNIST,Speed,Tl_y,P_y) :-
    Tl_y = -1;
    (Yr is P_y - Tl_y,
    (Yr < 0
    ;
    (Yr < 300,
    brake_dist(Yb,Speed),
    idle_dist(Yi,Speed),
    Yr >= Yb,
    Yr >= Yi)
    ;
    cell_traffic_light(Img,C),
    C = green)).


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace

action(Img,MNIST,Speed,Tl_y,P_y,1) :-
    speed_zone_brake(MNIST,Speed);
    traffic_light_brake(Img,MNIST,Speed,Tl_y,P_y);
    ped_brake(Img,Speed).
action(Img,MNIST,Speed,Tl_y,P_y,2) :-
    (traffic_light_idle(Img,MNIST,Speed,Tl_y,P_y);
    ped_idle(Img,Speed)),
    \+ action(Img,MNIST,Speed,Tl_y,P_y,1).
action(Img,MNIST,Speed,Tl_y,P_y,3) :-
    (speed_zone_follow(MNIST,Speed);
    ped_follow(Img,Speed)),
    \+ action(Img,MNIST,Speed,Tl_y,P_y,1),
    \+ action(Img,MNIST,Speed,Tl_y,P_y,2).
action(Img,MNIST,Speed,Tl_y,P_y,0) :-
    (traffic_light_accelerate(Img,MNIST,Speed,Tl_y,P_y);
    ped_acc(Img,Speed)),
    \+ action(Img,MNIST,Speed,Tl_y,P_y,1),
    \+ action(Img,MNIST,Speed,Tl_y,P_y,2),
    \+ action(Img,MNIST,Speed,Tl_y,P_y,3).
