import pandas as pd
import fire
import matplotlib.pyplot as plt
import matplotlib

COLORS = list(matplotlib.colors.TABLEAU_COLORS.values())

def make_sub_chart(batch_size_idx, techniques, df, ax, title, category_column, value_column, ylim_low, ylim_high, data_format, label, va):
    x_values = []
    y_values = []
    bar_colors = []
    x_idx = 0
    for key in techniques.keys():
        if key in df[category_column].tolist():
            x_values.append(key)
            y_values.append(df[value_column].tolist()[x_idx])
            bar_colors.append(COLORS[batch_size_idx])
            x_idx += 1
        else:
            x_values.append(key)
            y_values.append(0)
    x_coords = []
    for name in df[category_column]:
        if name in techniques:
            x_coords.append(techniques[name])
    ax.bar(x_values, y_values, label=label, color=bar_colors)

    # Customize the chart labels and title
    ax.set_xlabel(category_column)
    ax.set_ylabel(value_column)
    ax.set_title(title)
    if ylim_low is None:
        assert ylim_high is None
    else:
        ax.set_ylim(ylim_low, ylim_high)

    tick_positions = ax.get_yticks()
    for tick in tick_positions:
        ax.axhline(y=tick, color='gray', linestyle='--', alpha=0.7)

    # Add data labels or data points above the bars
    for x, value in zip(x_coords, df[value_column]):
        ax.text(x, value, data_format.format(value), ha='center', va=va)


def make_row_chart(batch_size_idx, techniques, df, value_column, ax1, ax2, label, ylim_low, ylim_high, va, title="", relative=False, data_format=None):
    category_column = "technique"
    if not isinstance(ylim_low, tuple):
        ylim_low = (ylim_low, ylim_low)
    if not isinstance(ylim_high, tuple):
        ylim_high = (ylim_high, ylim_high)

    def helper(sam_model_type, ax1, ylim_low, ylim_high, va):
        vit_b_df = df[df['sam_model_type'] == sam_model_type]

        vit_b_df = vit_b_df.copy()

        if relative:
            vit_b_df[value_column] = vit_b_df[value_column].div(
                vit_b_df[value_column].iloc[0])

        make_sub_chart(batch_size_idx, techniques, vit_b_df, ax1, f"{title} for {sam_model_type}",
                       category_column, value_column, ylim_low, ylim_high, data_format, label, va)
    helper("vit_b", ax1, ylim_low[0], ylim_high[0], va)
    helper("vit_h", ax2, ylim_low[1], ylim_high[1], va)

def run(csv_file,
        fig_format):
    matplotlib.rcParams.update({'font.size': 12})
    
    mdf_ = pd.read_csv(csv_file)
    mdf = mdf_.dropna(subset=["batch_size"])
    techniques = {'fp32': 0, 'bf16': 1, 'compile': 2, 'SDPA': 3, 'Triton': 4, 'NT': 5, 'int8': 6, 'sparse': 7}
    print("techniques: ", techniques)
    
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    
    for batch_size_idx, (batch_size, hlim, va) in enumerate(zip([32, 8], [100, 100], ["bottom", "top"])):
        df = mdf[mdf["batch_size"] == batch_size]
        make_row_chart(batch_size_idx, techniques, df, "img_s(avg)", *axs[0], f"Batch size {batch_size}", (0.0, 0.0), (100.0, 25.0), va,
                       "Images per second", data_format="{:.2f}")
        make_row_chart(batch_size_idx, techniques, df, "memory(MiB)", *axs[1], f"Batch size {batch_size}", 0, 40000, va,
                       title="Memory savings", data_format="{:.0f}")
        make_row_chart(batch_size_idx, techniques, df, "mIoU", *axs[2], f"Batch size {batch_size}", 0.0, 1.0, va,
                       title="Accuracy", data_format="{:.2f}")
    for ax in axs:
        ax[0].legend()
        ax[1].legend()
    # plt.tick_params(axis='both', which='both', length=10)
    plt.tight_layout()
    
    fig.savefig(f'bar_chart.{fig_format}', format=fig_format)

if __name__ == '__main__':
    fire.Fire(run)
