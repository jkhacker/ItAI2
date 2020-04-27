#!/usr/bin/env python
from PIL import Image
from random import randrange, random as rnd
import numpy as np
from copy import deepcopy
import cairo
from math import ceil
import sys
from numba import jit
import matplotlib.pyplot as plt
import imageio
from time import time


def overall_arrangement(creatures: list):
    o_fit = 0
    delta_fit = [
        (-(100 * (creatures[c].fitness / max_fit) - 100.0), c) for c in range(len(creatures)) if
        creatures[c].fitness < max_fit
    ]
    for i in range(len(delta_fit)):
        delta_fit[i] = (delta_fit[i][0] - delta_fit[-1][0], delta_fit[i][1])
    for i in range(len(delta_fit)):
        o_fit += delta_fit[i][0]
        overall_part = 0
        res_len = i
        for j in range(0, i + 1):
            part = ceil(pop_len * (delta_fit[j][0] / o_fit))
            overall_part += part
        if overall_part > pop_len:
            break
    o_fit -= delta_fit[res_len][0]
    res = []
    for i in range(res_len):
        res.append((ceil(pop_len * (delta_fit[i][0] / o_fit)), delta_fit[i][1]))
    return res


def key(obj):
    return obj.fitness


def bogosort(arr):
    new_arr = []
    while len(arr) != 0:
        new_arr.append(arr.pop(randrange(0, len(arr))))
    return new_arr


def progressbar(it, size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%f [%s%s] %i/%i\r" % (dif, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


@jit(nopython=True)
def calculate_fitness(A, B):
    # return (np.square(A - B)).mean()
    # return np.sum(np.sum((A - B)**2, axis=2)**0.5)
    return np.sum(np.sqrt(np.sum(np.square(np.subtract(A, B)), axis=2)))


class Creature:

    ischild = False

    def __init__(self, figs: list = []):
        self.data = np.zeros((512, 512, 4), dtype=np.uint8)
        self.surface = cairo.ImageSurface.create_for_data(self.data, cairo.FORMAT_ARGB32, 512, 512)
        self.context = cairo.Context(self.surface)
        self.figs = list(figs)
        if len(self.figs) == 0:
            self.fitness = max_fit
        else:
            self.ischild = True
            for fig in self.figs:
                self.context.set_source_surface(fig)
                self.context.paint()
            self.fitness = calculate_fitness(self.data, src_data)

    def generate_random_fig(self):
        s = cairo.ImageSurface(cairo.Format.ARGB32, 512, 512)
        cr = cairo.Context(s)
        cr.scale(512, 512)
        cr.move_to(rnd(), rnd())
        cr.rel_line_to(self.mutation_range(), self.mutation_range())
        cr.rel_line_to(self.mutation_range(), self.mutation_range())
        r, g, b, a = colors[randrange(0, len(colors))]
        cr.set_source_rgba(b, g, r, a)
        cr.fill_preserve()
        return s

    def mutation_range(self):
        alpha = ((100 - dif) / 100)
        return (rnd() - 0.5) * alpha

    def mutate(self):
        self.ischild = False
        fig = self.generate_random_fig()
        self.context.set_source_surface(fig)
        self.context.paint()
        # self.figs.append(fig)
        self.fitness = calculate_fitness(self.data, src_data)

    def crossover(self, dad):
        child_figs = bogosort(self.figs + dad.figs)
        for i in range(len(child_figs) // 2):
            child_figs.pop()
        return Creature(child_figs)

    def is_child(self):
        return self.ischild

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        setattr(result, 'data', deepcopy(self.data))
        setattr(result, 'fitness', deepcopy(self.fitness))
        setattr(result, 'figs', list(self.figs))
        setattr(
            result,
            'surface',
            cairo.ImageSurface.create_for_data(result.data, cairo.FORMAT_ARGB32, 512, 512))
        setattr(result, 'context', cairo.Context(result.surface))
        return result


if len(sys.argv) < 3:
    print("""
    Usage: ./main.py [image] [opacity] <optional> [number_of_generations]
    """)
filename = sys.argv[1]
opacity = float(sys.argv[2])
gen_file = 'gen_' + filename.split('.')[0] + '.png'
gen_gif = 'gen_' + filename.split('.')[0] + '.gif'
gen_len = 5000
if len(sys.argv) == 4:
    gen_len = int(sys.argv[3])



im_src = Image.open(filename, 'r').convert('RGBA')
colors = []
for r, g, b, a in list(im_src.getdata()):
    colors.append((r/255, g/255, b/255, opacity))
src_data = np.array(im_src).astype(np.uint16)

max_fit = calculate_fitness(np.zeros((512, 512, 4), dtype=np.uint8), src_data)
pop_len = 20

past_gen = [Creature() for i in range(pop_len)]
gif = imageio.get_writer(gen_gif, mode='I')
graph = []
dif = 0.0
i = 0
overall_timing = []
children_generation_timing = []
children_in_gen = []
now = time()

for i in progressbar(range(i, gen_len)):
    start_gen = time()
    new_gen = deepcopy(past_gen)
    # children = []

    # Mutations
    for being in new_gen:
        being.mutate()

    # Concatenation with the past generation
    new_gen += past_gen
    new_gen = sorted(new_gen, key=key)
    # start_children = time()
    # for j in range(1, 11):
    #     children.append(new_gen[j-1].crossover(new_gen[j]))
    # children_generation_timing.append(time() - start_children)
    # new_gen += children
    # new_gen = sorted(new_gen, key=key)

    # Calculating parts for each creature in new generation
    arrangements = overall_arrangement(new_gen)
    past_gen = []
    for part, index in arrangements:
        past_gen += [deepcopy(new_gen[index]) for i in range(part)]

    # Calculating difference with source picture
    dif = -(100*(past_gen[0].fitness/max_fit) - 100.0)
    graph.append(dif)

    # Saving current result
    if i % 100 == 0:
        Image.fromarray(past_gen[0].data).save(gen_file)
        image = imageio.imread(gen_file)
        gif.append_data(image)
    if dif > 99.99:
        break
    # children_in_gen.append(0)
    # for being in past_gen:
    #     if being.is_child():
    #         children_in_gen[-1] += 1
    # overall_timing.append(time()-start_gen)

print('\nOverall time: ', time()-now)

# plt.plot(list(range(len(graph))), graph)
# plt.xlabel('Generation')
# plt.ylabel('Fitness (in %)')
# plt.savefig('fitness.png')
# plt.close()
#
# plt.plot(list(range(len(overall_timing))), overall_timing)
# plt.xlabel('Generation')
# plt.ylabel('Calc time')
# plt.savefig('overall_timing.png')
# plt.close()
#
# plt.plot(list(range(len(children_generation_timing))), children_generation_timing)
# plt.xlabel('Generation')
# plt.ylabel('Children generation calc time')
# plt.savefig('children_timing.png')
# plt.close()
#
# plt.plot(list(range(len(children_in_gen))), children_in_gen)
# plt.xlabel('Generation')
# plt.ylabel('Children in generation')
# plt.savefig('children_in_gen.png')
# plt.close()





