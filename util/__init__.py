import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, inv

def fit_ellipse(x, y):
    """
    Fit an ellipse to the given x, y points using Fitzgibbon's direct least squares method.
    Returns ellipse parameters: center, axis lengths, rotation angle.
    Equation of ellipse: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    """
    # Build design matrix
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    # Scatter matrix
    S = np.dot(D.T, D)
    # Constraint matrix
    C = np.zeros((6,6))
    C[0,2] = C[2,0] = 2
    C[1,1] = -1

    # Solve generalized eigenvalue problem
    eig_vals, eig_vecs = eig(S, C)
    # Find the eigenvector with a positive eigenvalue that satisfies the ellipse constraint
    cond = np.logical_and(np.isreal(eig_vals), eig_vals > 0)
    a = np.real(eig_vecs[:, cond][:, 0])

    # Extract parameters
    A, B, C_coef, D_coef, E_coef, F_coef = a

    # Compute center of the ellipse
    denom = B**2 - 4*A*C_coef
    x0 = (2*C_coef*D_coef - B*E_coef) / denom
    y0 = (2*A*E_coef - B*D_coef) / denom

    # Compute the orientation and axes lengths
    term = np.sqrt((A - C_coef)**2 + B**2)
    # Semi-axes lengths
    numerator = 2*(A*E_coef**2 + C_coef*D_coef**2 + F_coef*B**2 - B*D_coef*E_coef - 4*A*C_coef*F_coef)
    denom1 = (B**2 - 4*A*C_coef)*( (C_coef + A) + term )
    denom2 = (B**2 - 4*A*C_coef)*( (C_coef + A) - term )
    a_len = np.sqrt(numerator/denom1)
    b_len = np.sqrt(numerator/denom2)

    # Compute rotation angle (in radians)
    if B == 0 and A < C_coef:
        theta = 0
    elif B == 0 and A >= C_coef:
        theta = np.pi/2
    else:
        theta = 0.5 * np.arctan(B/(A - C_coef))

    return (x0, y0), (a_len, b_len), theta

def ellipse_points(center, axes, angle, num_points=100):
    """
    Generate points on an ellipse.
    center: (x0, y0)
    axes: (a, b) lengths
    angle: rotation angle in radians
    """
    t = np.linspace(0, 2*np.pi, num_points)
    ellipse = np.array([axes[0]*np.cos(t), axes[1]*np.sin(t)])
    # Rotate
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    ellipse_rot = np.dot(R, ellipse)
    ellipse_rot[0] += center[0]
    ellipse_rot[1] += center[1]
    return ellipse_rot, t

def compute_angles(points, center, angle):
    """
    For each point, compute an angular coordinate along the ellipse.
    We first transform the point into the ellipse’s coordinate system (undo the rotation and center).
    Then the angle (atan2) in that system is taken as the ordering coordinate.
    """
    # Build rotation matrix to align with ellipse axes
    R = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    angles = []
    for pt in points:
        p = pt - center
        p_rot = R.dot(p)
        ang = np.arctan2(p_rot[1], p_rot[0])
        # Normalize angle to [0, 2pi)
        if ang < 0:
            ang += 2*np.pi
        angles.append(ang)
    return np.array(angles)

# --- Example usage ---

# Suppose you have some 2D points that roughly form an ellipse:
np.random.seed(42)
# Create an ellipse with center (10, 5), axes lengths 8 and 3, rotated by 30 degrees.
true_center = np.array([10, 5])
true_axes = (8, 3)
true_angle = np.deg2rad(30)

# Parameterize the ellipse
t_true = np.linspace(0, 2*np.pi, 300)
ellipse = np.array([true_axes[0]*np.cos(t_true), true_axes[1]*np.sin(t_true)])
R_true = np.array([[np.cos(true_angle), -np.sin(true_angle)],
                   [np.sin(true_angle),  np.cos(true_angle)]])
ellipse_rot = R_true.dot(ellipse)
ellipse_rot[0] += true_center[0]
ellipse_rot[1] += true_center[1]

# Sample some points from the ellipse and add noise
indices = np.random.choice(ellipse_rot.shape[1], size=50, replace=False)
points = ellipse_rot[:, indices].T + np.random.normal(scale=0.5, size=(50,2))

# Separate x and y
x = points[:,0]
y = points[:,1]

# Fit an ellipse to these points
fitted_center, fitted_axes, fitted_angle = fit_ellipse(x, y)
print("Fitted center:", fitted_center)
print("Fitted axes (semi-lengths):", fitted_axes)
print("Fitted rotation (radians):", fitted_angle)

# Generate points on the fitted ellipse for plotting
fitted_ellipse_pts, t_fit = ellipse_points(fitted_center, fitted_axes, fitted_angle, num_points=200)

# Order the original points along the ellipse:
angles = compute_angles(points, fitted_center, fitted_angle)
sort_order = np.argsort(angles)
ordered_points = points[sort_order]

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(fitted_ellipse_pts[0], fitted_ellipse_pts[1], 'r--', label='Fitted Ellipse')
plt.scatter(x, y, c='b', label='Original Points')
plt.scatter(ordered_points[:,0], ordered_points[:,1], 
            c=angles[sort_order], cmap='viridis', s=80, label='Ordered Points')
for i, pt in enumerate(ordered_points):
    plt.text(pt[0]+0.1, pt[1]+0.1, str(i), fontsize=9)
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Ellipse Fit and Point Ordering")
plt.axis('equal')
plt.show()
