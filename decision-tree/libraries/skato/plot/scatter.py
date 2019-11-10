# -*- coding: utf-8 -*-


def scatter_plot(one_column, second_column, **kwargs):
    output_name = "%s_and_%s_scatter_plot.png" % (one_column, second_column)
    output_path = os.path.join(kwargs["output_dir"], output_name)
    logging.info("Saving to %s" % output_path)
    g = sns.FacetGrid(dialogues[[one_column, second_column, "CORPUS"]],
                      col="CORPUS", col_wrap=2, size=10, aspect=1.0)
    g = g.map(plt.scatter, one_column, second_column, edgecolor="w")
    plt.savefig(output_path)
    plt.clf()
    logging.info("Saved to %s" % output_path)
