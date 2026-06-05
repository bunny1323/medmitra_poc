import secrets
def main():
    print(f"INTERNAL_API_KEY=medmitra_int_{secrets.token_urlsafe(32)}")
if __name__ == "__main__": main()

