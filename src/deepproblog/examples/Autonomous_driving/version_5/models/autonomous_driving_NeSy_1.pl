% Perception
nn(perc_net_version_5_NeSy_danger_pedestrian,[Img,Speed],X,[0,1,2,3]) :: cell_danger_pedestrian(Img,Speed,X).
nn(perc_net_version_5_NeSy_speed_zone,[MNIST],SZ,[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]) :: cell_speed_zone(MNIST,SZ).
nn(perc_net_version_5_NeSy_traffic_light,[Img],X,[green, orange, red]) :: cell_traffic_light(Img,X).
nn(perc_net_version_5_NeSy_intersection,[Img,P],X,[nothing,danger]) :: cell_danger_intersection(Img,P,X).
nn(perc_net_version_5_NeSy_traffic_sign,[Img],X,[priority_right,priority_all,priority_intersection]) :: cell_traffic_sign(Img,X).


%%%%    TRAFFIC RULES    %%%%

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
    \+ traffic_sign_valid(Img),
    Speed > 0,
    Margin = 85,
    Yr is P_y - Tl_y,
    Yr >= 0,
    brake_dist(Yb,Speed),
    Yb_margin is Yb + Margin,
    Yr < Yb_margin,
    Mb is Margin - 1,
    (Speed > 0.2 ; Yr < Mb),
    cell_traffic_light(Img,C),
    C = red.


traffic_light_idle(Img,MNIST,Speed,Tl_y,P_y) :-
    \+ Tl_y = -1,
    \+ traffic_sign_valid(Img),
    Margin = 85,
    Yr is P_y - Tl_y,
    Yr >= 0,
    Yr < 300,
    ((brake_dist(Yb,Speed),
    Yb_margin is Yb + Margin,
    Yr < Yb_margin,
    Mb is Margin - 1,
    (Speed =< 0.2 ; Yr >= Mb),
    cell_traffic_light(Img,C),
    C = red)).


traffic_light_accelerate(Img,MNIST,Speed,Tl_y,P_y) :-
    \+ Tl_y = -1,
    \+ traffic_sign_valid(Img),
    (
    (Yr is P_y - Tl_y + 85,
    (Yr < 0
    ;
    (Yr < 300,
    brake_dist(Yb,Speed),
    idle_dist(Yi,Speed),
    Yr >= Yb,
    Yr >= Yi)
    ;
    cell_traffic_light(Img,C),
    C = green))).


% Rules - intersection
intersection_region(left,P) :- P = 0.0.
intersection_region(right,P) :- P = 1.0.

intersection_brake(Img,Speed,Player_y,Intersection_y) :-
    Margin = 20,
    Speed > 0,
    Yr is Player_y - Intersection_y,
    Yr >= 0,
    Yr =< 300,
    brake_dist(Yb,Speed),
    Yb_margin is Yb + Margin,
    Yr < Yb_margin,
    Mb is Margin - 1,
    (Speed > 0.2 ; Yr < Mb),
    (traffic_sign_priority_right_brake(Img);traffic_sign_priority_all_brake(Img)).

intersection_idle(Img,Speed,Player_y,Intersection_y) :-
    Margin = 20,
    Yr is Player_y - Intersection_y,
    Yr >= 0,
    Yr =< 300,
    ((brake_dist(Yb,Speed),
    Yb_margin is Yb + Margin,
    Yr < Yb_margin,
    Mb is Margin - 1,
    (Speed =< 0.2 ; Yr >= Mb),
    (traffic_sign_priority_right_brake(Img);traffic_sign_priority_all_brake(Img)))
    ;
    (idle_dist(Yi,Speed),
    Yi_margin is Yi + Margin,
    Yr < Yi_margin,
    (traffic_sign_priority_right_brake(Img);traffic_sign_priority_all_brake(Img)))).

intersection_accelerate(Img,Speed,Player_y,Intersection_y) :-
    Yr is Player_y - Intersection_y,
    (
    (Yr >= 0,
    Yr =< 300,
    brake_dist(Yb,Speed),
    idle_dist(Yi,Speed),
    Yr >= Yb,
    Yr >= Yi)
    ;
    traffic_sign_priority_intersection(Img)
    ;
    traffic_sign_priority_right_accelerate(Img)
    ;
    traffic_sign_priority_all_accelerate(Img)).


% Rules - traffic signs
traffic_sign_valid(Img) :-
    cell_traffic_light(Img,C),
    C = orange.

traffic_sign_priority_right_brake(Img) :-
    traffic_sign_valid(Img),
    intersection_region(right,Pr),
    cell_traffic_sign(Img,priority_right),
    cell_danger_intersection(Img,Pr,danger).

traffic_sign_priority_right_accelerate(Img) :-
    traffic_sign_valid(Img),
    intersection_region(right,Pr),
    cell_traffic_sign(Img,priority_right),
    cell_danger_intersection(Img,Pr,nothing).

traffic_sign_priority_all_brake(Img) :-
    traffic_sign_valid(Img),
    intersection_region(left,Pl),
    intersection_region(right,Pr),
    cell_traffic_sign(Img,priority_all),
    (cell_danger_intersection(Img,Pl,danger);cell_danger_intersection(Img,Pr,danger)).

traffic_sign_priority_all_accelerate(Img) :-
    traffic_sign_valid(Img),
    intersection_region(left,Pl),
    intersection_region(right,Pr),
    cell_traffic_sign(Img,priority_all),
    (cell_danger_intersection(Img,Pl,nothing),cell_danger_intersection(Img,Pr,nothing)).

traffic_sign_priority_intersection(Img) :-
    traffic_sign_valid(Img),
    cell_traffic_sign(Img,priority_intersection).


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace

action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,1) :-
    speed_zone_brake(MNIST,Speed);
    traffic_light_brake(Img,MNIST,Speed,Tl_y,Player_y);
    intersection_brake(Img,Speed,Player_y,Intersection_y);
    (Speed > 0, cell_danger_pedestrian(Img,Speed,3)).

action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,2) :-
    \+ cell_traffic_light(Img, green),
    (traffic_light_idle(Img,MNIST,Speed,Tl_y,Player_y);
    intersection_idle(Img,Speed,Player_y,Intersection_y);
    (cell_danger_pedestrian(Img,Speed,2);
    Speed = 0, cell_danger_pedestrian(Img,Speed,3))),
    \+ action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,1).

action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,3) :-
    (speed_zone_follow(MNIST,Speed);
    cell_danger_pedestrian(Img,Speed,1)),
    \+ action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,1),
    \+ action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,2).

action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,0) :-
    (traffic_light_accelerate(Img,MNIST,Speed,Tl_y,Player_y);
    intersection_accelerate(Img,Speed,Player_y,Intersection_y);
    cell_danger_pedestrian(Img,Speed,0);
    speed_zone_accelerate(MNIST,Speed)),
    \+ action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,1),
    \+ action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,2),
    \+ action(Img,MNIST,Speed,Player_y,Intersection_y,Tl_y,3).
