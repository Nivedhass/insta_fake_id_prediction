import instaloader

def scrape_instagram(username):
    L = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        
        # Extract features
        data = {
            "profile pic": 1 if profile.profile_pic_url else 0,
            "nums/length username": len(username) / max(len(username), 1),
            "fullname words": len(profile.full_name.split()),
            "nums/length fullname": sum(c.isdigit() for c in profile.full_name) / max(len(profile.full_name), 1),
            "name==username": int(profile.full_name.lower() == username.lower()),
            "description length": len(profile.biography),
            "external URL": int(profile.external_url is not None),
            "private": int(profile.is_private),
            "#posts": profile.mediacount,
            "#followers": profile.followers,
            "#follows": profile.followees
        }
        
        return data
    except Exception as e:
        return {"error": str(e)}
