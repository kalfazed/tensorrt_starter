-- For new users:
-- When you run lvim firstly, please run :PackerInstall and :PackerCompile to compile the packages
-- Then a "packer_compild.lua" will be created under plugin folder
-- Please make sure the neovim version is v0.8.x
-- For a list of keyboard shortcut, please type <space> <s> <k>
-- Have fun!!

reload "user.telescope"
reload "user.keymaps"
reload "user.treesitter"
reload "user.lsp"
reload "user.nvimtree"
reload "user.plugins"
reload "user.signature"
reload "user.outlines"
reload "user.autocommands"
reload "user.chatgpt"


-- general 
-- vim.format_on_save.enabled = false
lvim.colorscheme = "tokyonight-moon"
lvim.transparent_window = true
vim.opt.clipboard = "unnamedplus"

lvim.builtin.alpha.active = true
lvim.builtin.alpha.mode = "dashboard"
lvim.builtin.terminal.active = true
