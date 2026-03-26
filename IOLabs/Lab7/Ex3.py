import asyncio
from twscrape import API
import json

async def main():
    api = API()

    # add account to scrapping
    await api.pool.add_account("username", "password", "email", "mail password")
    await api.pool.login_all()

    async for tweet in api.search("spellforce", limit=100):
        dict_to_json = {"tweet id": tweet.id,
                        "user name": tweet.user.username,
                        "content": tweet.rawContent}

        json_object = json.dumps(dict_to_json)
        with open(f"Tweet {tweet.id}.json", 'w') as output:
            output.write(json_object)
        print(tweet.id, tweet.user.username, tweet.rawContent)

if __name__ == "__main__":
    asyncio.run(main())
