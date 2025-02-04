function optimal_input = optimal_pendulum(y0, target)

    theta1 = y0(1);
    omega1 = y0(2);
    theta2 = y0(3);
    omega2 = y0(4);

    % Parameters
    m1 = 1.0;
    m2 = 0.5;
    l1 = 1.0;
    l2 = 0.5;
    g = 9.81;
    max_input = 20.0;   

    % Calculate weight compensation
    component1 = l1 * m1 * g * sin(target);
    component2 = m2 * g * (l1 * sin(target) + l2 * sin(theta2));

    % Calculate feedback and velocity components
    feedback = target - theta1;
    component3 = 20 * feedback;
    component4 = -10 * omega1;

    % Compute and clip the optimal input
    optimal_input = min(max(component1 + component2 + component3 + component4, -max_input), max_input);
end
