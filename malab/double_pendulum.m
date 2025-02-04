function [t,y] = double_pendulum(tspan,y0, torque, damper)

    function result = equations_of_motion(t, y0)
        % Parameters
        m1 = 1.0;
        m2 = 0.5;
        l1 = 1.0;
        l2 = 0.5;
        g = 9.81;
        k = 0.0;
        max_input = 20.0;
        
        theta1 = y0(1);
        omega1 = y0(2);
        theta2 = y0(3);
        omega2 = y0(4);
    
        % Pre-compute terms for the equations
        delta = theta2 - theta1;
        M = m1 + m2;
        m2l1l2_cos_delta = m2 * l1 * l2 * cos(delta);
        m2l1l2_sin_delta = m2 * l1 * l2 * sin(delta);
    
        % Equations derived from Lagrangian mechanics
        a = (M * l1^2);
        b = m2l1l2_cos_delta;
        c = m2l1l2_cos_delta;
        d = m2 * l2^2;

        clip_torque = min(max(torque, -max_input), max_input);
    
        f1 = -m2l1l2_sin_delta * omega2^2 - M * g * l1 * sin(theta1) + clip_torque;
        f2 = m2l1l2_sin_delta * omega1^2 - m2 * g * l2 * sin(theta2);
    
        % Spring and damper forces
        spring_force = k * (delta);
        damper_force = damper * (omega2 - omega1);  
    
        % Add spring and damper forces to equations of motion
        f1 = f1 + spring_force + damper_force;
        f2 = f2 - (spring_force + damper_force);
    
        % Solving linear system for accelerations
        A = [a b; c d];
        b = [f1 f2]';
    
        res = linsolve(A, b);
    
        accel1 = res(1);
        accel2 = res(2);
        result = [omega1; accel1; omega2; accel2];
    end

    [t,y] = ode45(@equations_of_motion, tspan, y0);

end