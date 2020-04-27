import pstats

p = pstats.Stats('kurisu.stat')
p.sort_stats('cumulative').print_stats()