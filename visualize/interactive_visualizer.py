import argparse

from pyvis.network import Network


ACTION_COLOR = ["red", "blue", "green", "purple"]


def get_label(label):

    if type(label) == str:
        return label
    elif type(label) == int:
        return "%d" % label
    else:
        return "%r" % label


def show_graph(transitions, savefile, threshold):
    """
    Build a graph
    :param transitions: List of tuples [action, state, next_state_1:prob_1, next_state_2:prob_2, ....]
    :param threshold: value below which transition probabilities will not be visualized
    :return: network
    """

    net = Network(directed=True)

    state_seen = dict()
    action_seen = dict()

    for transition in transitions:

        words = [word.strip() for word in transition.split()]

        action = words[0]
        state = words[1]

        if state not in state_seen:
            state_seen[state] = len(state_seen) + 1
            net.add_node(state_seen[state],
                         label=get_label(state),
                         shape="circle")

        if action not in action_seen:
            action_seen[action] = ACTION_COLOR[len(action_seen)]

        next_state_probs = [next_state_prob.split(":") for next_state_prob in words[2:]]

        for next_state, prob in next_state_probs:

            if next_state not in state_seen:
                state_seen[next_state] = len(state_seen) + 1
                net.add_node(state_seen[next_state],
                             label=get_label(next_state),
                             shape="circle")

            prob = float(prob)
            if prob >= threshold:
                net.add_edge(state_seen[state], state_seen[next_state], width=prob * 5, color=action_seen[action])

        net.set_edge_smooth('dynamic')
        net.show(savefile)


def parse(fname):

    lines = open(fname).readlines()
    start, end = -1, -1
    data_receiving = False

    for ix, line in enumerate(lines):
        if len(line.strip()) == 0 and start == -1:
            start = ix
            continue

        if start != -1 and not line.startswith("#"):
            data_receiving = True

        if data_receiving and line.startswith("#"):
            end = ix
            break

    if end == -1:
        end = len(lines)

    return [line.strip() for line in lines[start + 1:end] if not line.startswith("#")]


parser = argparse.ArgumentParser()
parser.add_argument("--name", default="./discrete-factors/dp/example_output", help="Name of data file")
parser.add_argument("--save", default="./discrete-factors/visualize/nx.html", help="Name of save file")
parser.add_argument("--threshold", default=0.05, type=float,
                    help="Value below which transition probabilities will not be visualized")
args = parser.parse_args()

data = parse(args.name)
show_graph(transitions=data,
           savefile=args.save,
           threshold=args.threshold)
