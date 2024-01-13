# https://chat.openai.com/share/5f3f2019-2051-4217-93ea-c926fa3c2749
try:
    import markdownify
except ImportError:
    raise ImportError("Package 'markdownify' is not installed. Please install it using 'pip install markdownify'")

try:
    from IPython.core.getipython import get_ipython
    from IPython.display import HTML
except ImportError:
    raise ImportError("Package 'IPython' is not installed. Please install it using 'pip install ipython'")

try:
    from rich import print
    from rich.console import Console
    from rich.markdown import Markdown
except ImportError:
    raise ImportError("Package 'rich' is not installed. Please install it using 'pip install rich'")


def print_html(obj):
    # Create a Rich console that outputs to a string
    console = Console()

    # Convert the HTML to Markdown
    markdown = markdownify.markdownify(obj.data)
    rich_markdown = Markdown(markdown)
    console.print(rich_markdown)


if __name__ == "__main__":
    # Create InteractiveShell instance
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.instance()


# Get the current IPython instance
ipython = get_ipython()

if ipython is not None:
    # Register the renderer for HTML objects
    ipython.display_formatter.ipython_display_formatter.for_type(HTML, print_html)


if __name__ == "__main__":
    print(
        ipython.display_formatter.format(
            HTML(
                "<h1>Hello World</h1><p>This is a paragraph</p><ul><li>Item 1</li><li>Item 2</li></ul>"
            )
        )
    )
