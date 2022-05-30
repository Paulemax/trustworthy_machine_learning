# Task 1

## Methods

Saliency(Sal): \
$S_{i}^{f_c}(x) = \frac{\partial f_c(x)}{\partial x_i}$

Gradient times Input(GtI): \
$GxI_{i}^{f_c}(x) = x_i \frac{\partial f_c(x)}{\partial x_i}$

Integrated Gradients(IG): \
$IG_{i}^{f_c}(x) = \int_{0}^{1} d \alpha \frac{\partial f_c (\gamma(\alpha))}{\partial \gamma_i(\alpha)} \frac{\partial \gamma_i(\alpha)}{\partial \gamma}$ \
with $\gamma(0) = x' $ and $\gamma(1) = x$, if not otherwise specified otherwise we consider a straight path

## Part One

Check if IG reduces to: $(x_i - x_i') *\int_{0}^{1} \frac{\partial f_c(x' + \alpha \times (x - x'))}{\partial x_i}$ if \
$\gamma(\alpha) = x' + \alpha* (x - x')$ for $\alpha \in [0, 1] $

$IG_{i}^{f_c}(x) = \int_{0}^{1} d \alpha \frac{\partial f_c (x' + \alpha*(x - x'))}{\partial x' + \alpha* (x - x')} \frac{\partial x' + \alpha* (x - x')}{\partial(\gamma)}$

## Axioms

Completness: \
Sensitivity: \
Lineartiy: \

