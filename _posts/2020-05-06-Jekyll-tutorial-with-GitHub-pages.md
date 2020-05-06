---
layout: post
title: Jekyll tutorial with GitHub pages
author: Silver Surfer
date: '2017-09-13 14:35:23 +0530'
tags: tutorial git github
category:
        - guides
        - github
        - tutorials
comments: true
excerpt_separator: <!--more-->
thumbnail: github-page.png
---

There is a great satisfaction to set up own Blog on GitHub pages. It took me a while to gather all the information from various sources and combine it to create a stylish website.

This article will lead you through all the steps and explain how set up your own open, professional-looking Blog. Which is perfect to show off portfolio, share knowledge and post easy to access tech articles for your future self.
<!--more-->

You'll be surprised how simple it is to combine most powerful technologies these days!

The guide assumes the reader has little to no knowledge of html/css/javascript/ruby/github-commands or other languages. Anybody with PC should be able to accomplish it in a few hours.

Without further ado... **Let's get started!**


## Step 1: [GitHub Pages](https://pages.github.com/)

You'll need a github account. If you don't yet have one, [register it](github.com).

Quick question: **What is git?** - *Simply put, it is a control version manager.*

Ok, let's move on.

There is a Github service: [pages.github.com](https://pages.github.com/) conveniently allows to host a personal (or company) website with a hostname like: `username.github.io`
From here on you have 2 choices:

- Follow tutorial on amazing [pages.github](https://pages.github.com/) main page and create a page super quickly. The drawback is less customisation.
- Follow this guide further. Understand and apply configurations. Maintain website locally.

As github pages suggest, the very first steps would be:

> Head over to [GitHub]((github.com)) and create a new repository named username.github.io, where username is your username (or organization name) on GitHub.

![Create Repository username.github.io]({{ site.url }}/images/create_repo.png)

It is important to create repository with username exactly the same as registered. No need to initialize repository with `ReadMe.md`.

A terminal as a git client required for further steps. [Get latest version here.](https://git-scm.com/download/win)
Next item would be to configure authentication for connection to Git via Git terminal.

1. [Caching password in Git](https://help.github.com/articles/caching-your-github-password-in-git/) 
2. [Generate SSH key](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)

Cool!

*Brace yourself! And follow carefully step by step. It is rewarding after all :)*

A few basic commands to navigate through your system via the terminal:
`pwd` - shows the directory you are in
`ls` - lists files in current directory
`cd` - "change directory" - navigating to a specified directory

Now we'll fetch the repository to you machine by running the command (be sure to replace *username* with yours):

```
git clone https://github.com/username/username.github.io
```

This will create a directory named username.github.io inside whatever folder you are currently in. From the command name it is evident that the repository got "cloned" to your machine (yes, in fact an empty repository would be cloned and initialized locally)

Eventually this newly created local folder will house all `.html`, `.css`, `.js` and `.markdown` pages of your website. All articles and posts edited or created locally then should be pushed to git repository to be presented on the web.

## Step 2: Jekyll

As explained on Jekyll's [website](https://jekyllrb.com/):

>Jekyll is a simple, blog-aware, static site generator. It takes a template directory containing raw text files in various formats, runs it through a converter (like Markdown) and our Liquid renderer, and spits out a complete, ready-to-publish static website suitable for serving with your favorite web server. Jekyll also happens to be the engine behind GitHub Pages, which means you can use Jekyll to host your project’s page, blog, or website from GitHub’s servers for free.

To explain it even friendlier: Jekyll allows to make simple webpages just by writing content in templates (using Markdown language), which are automatically converted into beautiful HTML pages with mobile-friendly responsiveness. When put on github it generates a website that is hosted absolutely free!

**Sounds great. Lets install it!**

Jekyll requires ruby, so as a prerequisite we need to install Ruby and all necessary Gems:
[Get Ruby](https://www.ruby-lang.org/en/downloads/)
After ruby got installed, you'll need to open a new terminal window, to run ruby commands.
Let's ensure we have latest version modules, install jekyll and a bundler:

```
gem update --system
gem install jekyll
gem install bundler
bundler update
```

Great!
Now, cd to your folder `cd username.github.io` and create the website:

```
jekyll new .
jekyll serve
```


Several lines would be printed out, one of them would look like `http://localhost:4000`. Open it in browser.
Ta-da! Website is ready!

## Step 3: [GitHub](https://github.com/)

In this step we'll move entire website onto web and make it available at `http://username.github.io`.

Remember we ran the command `git clone https://github.com/username/username.github.io` - it not only cloned, but also primed our local git to a remote address (to a repository).

Just in case, alternative to `clone` would be commands:

```
git init
git remote add origin https://github.com/username/username.github.io
git pull origin master --allow-unrelated-histories
```

Now, when we create a post we want it to become available on web! It can be done using following commands.

First of all the pages should be built, using:

```
jekyll serve
```

Ensure it's been built and looks good locally. Then run following:

```
git add .
git commit -m "initial commit"
git push origin master
```

You will be prompted for password upon `push` to github.

That would be a usual set of commands when publishing.

## Step 4: Create a page in Jekyll

The structure of a blog is very simple.
Say, you want to create `About` (or _Portfolio_ or *Contacts*) section. Simply create `about.md` page in the top level of your website, i.e. directly in your `username.github.io` folder.
In case you want to write posts, create a file in `_posts` folder with naming convention as in the example: `2017-09-10-New-Post.markdown`.

Any text or code editor is suitable to manage `*.markdown` files. However I recommend [Typora](https://typora.io/) as absolutely amazing Markdown editor of choice. Check it out!

Let's create `About` page as an example.
Create a new file, called `about.md` in your `username.github.io` folder. `md` stands for Markdown.
The contents of the file would be like this:

```html
---
layout: page
title: about
comments: true
---
![Spike](../images/Spike.jpg 'See you, Space Cowboy!')
Hello, I'm Spike! [I travel the galaxy](https://myanimelist.net/anime/1/Cowboy_Bebop) catching for bounty constantly getting into some serious "adventures" with my absolutely crazy crew. *Be water, my friend* is the way of living that ensures I'm still alive.
```

That snippet is a good example of Markdown text used to create pages. Markdown simple and friendly syntax will be automatically converted into a full fledged html by Jekyll. The page would look like this:

---

![Spike]({{ site.url }}/images/Spike.jpg 'See you, Space Cowboy!')
Hello, I'm Spike! [Travelling the galaxy](https://myanimelist.net/anime/1/Cowboy_Bebop) catching for bounty and constantly getting into some serious "adventures" with my amazing and crazy crew. "*Be water, my friend*" is the way of living that ensures I'm still alive.

---

File starts with text between `---` lines - it is called Front Matter. That section contains some necessary details of the page, such as template, date, title.
Next goes actual body of the page. `![Spike](../folder/image.jpg)` for example inserts an image.
`[Something](https://...)` - creates an url
A variety of syntax highlights can be used. Here is a handy [Cheatsheet on Markdown](http://assemble.io/docs/Cheatsheet-Markdown.html).

## Step 5: Create a post

Creating a post or article is easy and similar to the previous step.

Ensure to:
1.  In the `_posts` folder create a Markdown file, name it as following: YYYY-MM-DD-post-name.md
2.  In Front Matter section, set the `layout` to post: `layout: post`
3.  Set date and title in Front Matter

Now, the blog post would appear on your main page!

#### Preloaded text

[Post excerpts](http://jekyllrb.com/docs/posts/#post-excerpts) are first lines of your article content displayed on the main page.

By default only the first paragraph will be displayed as an excerpt. Alternatively can be controlled by setting custom separator in the Front Matter section as following:

```
---
excerpt_separator: <!--more-->
---
```

## Step 6: Adding comments!

We'll use a popular service called [Disqus](https://disqus.com/home/), you probably familiar with so far. So, a Disqus account is required. Go get it, if you don't have one.

When we did `jekyll new <PATH>` command, Jekyll installed a website with *gem-based theme* called Minima.
What is so called *gem-based*? Simply put, most configs, templates and theme files are stored as gems for your Ruby.
As explained in [Jekyll docs](https://jekyllrb.com/docs/themes/) the goal is:
>To allow you to get all the benefits of a robust, continually updated theme without having all the theme’s files getting in your way and over-complicating what might be your primary focus: creating content.

Ok, what that means for us in practice, is that we need to copy certain files from gem folder to our `username.github.io` folder in order to edit it and add comments!
Find out where gem is stored on your computer with a command `bundle show minima`:

```
E:\Git\silversurfer0.github.io>bundle show minima
C:/Ruby24-x64/lib/ruby/gems/2.4.0/gems/minima-2.1.1
```

Jekyll docs explain:
>To replace layouts or includes in your theme, make a copy in your `_layouts` or `_includes` directory of the specific file you wish to modify, or create the file from scratch giving it the same name as the file you wish to override.

Ok, keeping above in mind, lets do the following:
Create `_includes` folder in  `username.github.io` directory. Then create a file `disqus_comments.html` inside, with the contents:

![if page comments](/images/pages_github_if_page_comments.png)

Any code in between these two lines will be included if Front Matter for that page has `comments: true`.
Go to [Disqus to get Universal Code](https://silverthread.disqus.com/admin/install/) for Jekyll and insert it between `if` lines.
Universal code looks like this:

```javascript
<div id="disqus_thread"></div>

<script>

/**

- RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
- LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
  /*
  var disqus_config = function () {
  this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
  this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
  };
  */
  (function() { // DON'T EDIT BELOW THIS LINE
  var d = document, s = d.createElement('script');
  s.src = 'https://silverthread.disqus.com/embed.js';
  s.setAttribute('data-timestamp', +new Date());
  (d.head || d.body).appendChild(s);
  })();
  </script>
  <noscript>Please enable JavaScript to view the comments powered by Disqus.</noscript>
```

Please note, that `s.src` is given for my website! Disqus should provide you with your own url.

Next, copy `post.html` template from gem-based `_layouts` to local `_layouts` folder.
In this layout template we'll put one-line code that would include disqus comments to any post we want. So, just above the closing `</article>` tag add the following:

![include disqus comments](/images/include_disqus_comments.png)

Cool!

Posts with `comments: true` set in the Front Matter should now have comments enabled!

#### Optional: **set up comment count**

Add the following to `_layouts/default.html` before the closing `body` tag. Change `SHORTNAME` to the Disqus shortname you are using.

```javascript
<script id="dsq-count-scr" src="//SHORTNAME.disqus.com/count.js" async></script>
```

Add `#disqus_thread` to the end of a URL and Disqus will count the comments on the page the link points to. For example, my `_layouts/post.html` contains the following code. Note the comment count at the top of this post.

![pages_comment_count](/images/pages_github_comment_count.png)

`index.html` contains the following code to display the comment count for each post in the home page:

```javascript
<a href="https://sgeos.github.io{{ post.url }}#disqus_thread">0 Comments</a>
```

Possible reference, in case of hostname or postname migration, a [Disqus Migration Tool](https://help.disqus.com/customer/portal/articles/286778-migration-tools) can be used, to transfer comments if any.

## Step 7: Adding Google Analytics for Jekyll

Google Analytics is a powerful tool to see how your site performs on the web with vast capabilities, all provided by a small code snippet to be inserted into the head section of a website. Go get [unique code snippet from Google Analytics](http://analytics.google.com/).

Analytics provides several options:
1. A new Website Tracking `gtag.js` (Global Site Tracking). New Beta.
2. `analytics.js` - a conservative choice. "*If you are not ready to upgrade to Website Tracking yet*" - as their page says.

I stick to the newest option. It looks something like this:

```javascript
<!-- Global Site Tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=<YOUR UNIQUE ID>"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments)};
  gtag('js', new Date());
  gtag('config', <YOUR UNIQUE ID>);
</script>
```

Create `google-analytics.html` in `_includes`, and insert Google Analytics script from above.
Next, we'll enable it on all pages. Doing this is easy: include the google-analytics into relevant template file.

Copy `default.html` template from gem-based `_layouts` to your local `_layouts` folder.
Put this line:

![include analytics](/images/pages_github_analystics.png)

 in between the opening `<html>` and `<body>` tags of the default.html file.
**Congratulations! We are done!**

## Conclusion

Hope this guide was helpful! We've gone through quite a lot of steps so far! This is rewarding!

Now you have a solid Blog!
From now on this powerful tool would be perfect to share articles, cheatsheets, show off portfolio and all this in a transparent and reachable form all over the world provided by amazing github!

Again, a handy cheatsheet of most frequent commands.
*First serve and build pages locally:*

```
bundle exec jekyll serve
```

*Then upload to github:*

```
cd username.github.io
git init
git clone https://github.com/username/username.github.io.git
git commit -m "Update with my new useful article on Machine Learning"
git push origin master
```

**Enjoy blogging!**

Your comments are highly regarded, whether it is just saying hi, weight in with appreciation, addition, edition, suggestion, rendition. Or any other form of "dition". 

																*See you, space cowboy...*

#### Sources
[Great tutorial by Brian Caffey](https://briancaffey.github.io/2016/03/17/jekyll-tutorial.html) - Thanks Brian!

[Sgeos for adding disqus](http://sgeos.github.io/jekyll/disqus/2016/02/14/adding-disqus-to-a-jekyll-blog.html)

[Jekyll docs](https://jekyllrb.com/docs/home/)