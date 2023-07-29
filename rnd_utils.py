import torch
from torch import nn


class RunningMeanStd(nn.Module):
    '''
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    '''
    def __init__(self, epsilon=1e-4, shape=()) -> None:
        super().__init__()

        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float))
        self.count = epsilon
    
    def update(self, x: torch.Tensor) -> None:
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        if batch_count == 1:
            batch_var = torch.zeros_like(batch_mean)
        
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int) -> None:

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        new_count = total_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
    
    @property
    def std(self):
        return torch.sqrt(self.var)


if __name__ == "__main__":
    rsd = RunningMeanStd()
    print(rsd.mean + torch.rand(2, 3), rsd.var + torch.rand(2, 3))
