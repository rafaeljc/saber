import subprocess

def run():
    """Run the application from the command line.
    
    Enables the user to start the application by running the 'saber' command in 
    the terminal after installing it with pip.

    Usage example:
        $ pip install .
        $ saber
    """
    try:
        subprocess.run(["python", __file__])
    except KeyboardInterrupt:
        print("Application stopped by user.")

def main():
    print("Hello, I'm S.A.B.E.R.!")

if __name__ == "__main__":
    main()
