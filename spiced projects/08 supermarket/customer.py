
class Customer:

    """
    a single customer that moves through the supermarket
    in a MCMC simulation
    """
    def __init__(self, id, state, name, budget=100):
        self.name = name
        self.id = id
        self.state = state
        self.budget = budget

    def __repr__(self):
        return f'ID:{self.id} {self.name} is at {self.state} with a budget of {self.budget}'


if __name__ == "__main__":
    instance = Customer('1','dairy')
    print(instance.id)
    print(instance.name)
    print(instance.state)
    print(instance.budget)
    print(instance)