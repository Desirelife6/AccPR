

/**
 * Copyright (C) SEI, PKU, PRC. - All Rights Reserved.
 * Unauthorized copying of this file via any medium is
 * strictly prohibited Proprietary and Confidential.
 * Written by Jiajun Jiang<jiajun.jiang@pku.edu.cn>.
 */
package cofix.core.parser.search;

//import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.JavaParser;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

// import mfix.common.util.JavaFiles;
// import mfix.core.node.ast.MethDecl;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.AnonymousClassDeclaration;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.Statement;
import org.eclipse.jdt.core.dom.Type;

import cofix.common.config.Constant;
import cofix.common.util.JavaFile;
import cofix.common.util.Pair;
import cofix.core.match.CodeBlockMatcher;
import cofix.core.metric.CondStruct;
import cofix.core.metric.MethodCall;
import cofix.core.metric.OtherStruct;
import cofix.core.metric.Variable;
import cofix.core.parser.NodeUtils;
import cofix.core.parser.ProjectInfo;
import cofix.core.parser.node.CodeBlock;

import static cofix.common.config.Constant.HOME;

/**
 * @author Jiajun
 * @date Jun 29, 2017
 */
public class SimpleFilter {
    private List<CodeBlock> _candidates = new ArrayList<>();
    private CodeBlock _buggyCode = null;
    private Set<Variable> _variables = null;
    private Set<CondStruct.KIND> _condStruct = null;
    private Set<OtherStruct.KIND> _otherStruct = null;
    private Set<String> _methods = null;
    private int _max_line = 0;
    private int DELTA_LINE = 10;


    public SimpleFilter(CodeBlock buggyCode) {
        _buggyCode = buggyCode;
        _variables = new HashSet<>(buggyCode.getVariables());
        _condStruct = new HashSet<>();
        for(CondStruct condStruct : buggyCode.getCondStruct()){
            _condStruct.add(condStruct.getKind());
        }
        _otherStruct = new HashSet<>();
        for(OtherStruct otherStruct : buggyCode.getOtherStruct()){
            _otherStruct.add(otherStruct.getKind());
        }
        _methods = new HashSet<>();
        for(MethodCall call : _buggyCode.getMethodCalls()){
            _methods.add(call.getName());
        }
        _max_line = _buggyCode.getCurrentLine() + DELTA_LINE;
    }
    public List<Pair<CodeBlock, Double>> vectorFilter(String srcPath, String projectName, String bugId, double guard) throws IOException {
        List<String> files = JavaFile.ergodic(srcPath, new ArrayList<String>());  //从src文件读取文件路径
        List<Pair<CodeBlock, Double>> filtered = new ArrayList<>();
        String base_url = HOME + "/result/" +projectName + "/" +bugId;
        /*
        String method = "public void test(){\r";  //处理数据
        CollectorVisitor collectorVisitor = new CollectorVisitor();
        //处理数据
        File writesrc = new File(HOME+"/clone/data_prediction/java/src.tsv");
        File writepkl = new File(HOME+"/clone/data_prediction/java/lables.csv");
        BufferedWriter writeTextsrc = null;
        BufferedWriter writeTextpkl = null;
        try {
            writeTextsrc = new BufferedWriter(new FileWriter(writesrc));
            writeTextpkl = new BufferedWriter(new FileWriter(writepkl));
            writeTextpkl.write("id1"+","+"id2"+","+"label");
            writeTextsrc.write("1"+"\t"+"\""+method+_buggyCode.toSrcString().toString().replace('\n','\r')+"}"+"\"");
        } catch (IOException e) {
            e.printStackTrace();
        }
        */

        String method = "public void test(){ ";
        String end = "}";
        String code = (method+_buggyCode.toSrcString().toString()).replace('\n', ' ')+end;
        JavaParser j = new JavaParser();
        com.github.javaparser.ast.CompilationUnit cu;
        cu = j.parse(code).getResult().get();
        //String result = "";
        //String tmp = "";
        //File writesrc = new File(HOME + "/codebert/data/src.txt");
        File writesrc = new File(base_url + "/src.txt");
        BufferedWriter writeTextsrc = new BufferedWriter(new FileWriter(writesrc));
        cu.getTokenRange().get().forEach(t->{
            try {
                String str = "";
                String result = "";
                str = t.getText().replaceAll("\\s*", "");
                if(!str.equals(""))
                    result = result + str + " ";
                writeTextsrc.write(result);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        CollectorVisitor collectorVisitor = new CollectorVisitor();
        //File writesrc = new File(HOME + "/codebert/data/src.txt");
//        try {
//            writeTextsrc = new BufferedWriter(new FileWriter(writesrc));
//            writeTextsrc.write((method+_buggyCode.toSrcString().toString()).replace('\n', ' ')+end);
//            writeTextsrc.write(result);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        // 遍历文件，待搜索用于修复的代码库，不用改
        for(String file : files){
            CompilationUnit unit = null;
            try{
                unit = JavaFile.genASTFromFile(file);
            }catch(Exception e){
                continue;
            }
            collectorVisitor.setUnit(file, unit);
            unit.accept(collectorVisitor);
            filtered = codefilter(filtered);

        }

        if(filtered.size()==0){
            return new ArrayList<>();
        }
        List<String> exist = new ArrayList<>();//filter，存储去重后的代码段的字符串
        List<Pair<CodeBlock, Double>> match = new ArrayList<>();  //存储<codeblock，similarity>二元组
        for(Pair<CodeBlock, Double> pair : filtered){  //将codeblock改为string类型存入exist和match
            if(exist.contains(pair.getFirst().toSrcString().toString())){
                continue;
            }
            exist.add(pair.getFirst().toSrcString().toString());
            match.add(pair);
        }

        for(int i = 0; i < exist.size(); i++) {
            try {
                writeTextsrc.newLine();
                String filteredCode = (method+exist.get(i)).replace('\n', ' ')+end;
                cu = j.parse(filteredCode).getResult().get();
                cu.getTokenRange().get().forEach(t->{
                    try {
                        String str = "";
                        String result = "";
                        str = t.getText().replaceAll("\\s*", "");
                        if(!str.equals(""))
                            result = result + str + " ";
                        writeTextsrc.write(result);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
                //writeTextsrc.write((method+exist.get(i)).replace('\n', ' ')+end);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            writeTextsrc.flush();
            writeTextsrc.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        File wd = new File("/bin");
        System.out.println(wd);
        Process proc = null;
        try {
            proc = Runtime.getRuntime().exec("/bin/sh", null, wd);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (proc != null) {
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(proc.getOutputStream())), true);
            //System.out.println("-------codebert-------");
            out.println("conda activate codeBert");
            //out.println("cd /Users/seventeen/Documents/2-Research/sever_simfix/Simfix/codebert");
            out.println("cd /data/Simfix/codebert");
            out.println("pwd");
            out.println("/root/anaconda3/envs/codeBert/bin/python /data/Simfix/codebert/dataProcess_final.py --project_name=" + projectName + "--bug_id=" + bugId);
            //out.println("/Users/seventeen/opt/anaconda3/envs/codebert/bin/python dataProcess_final.py");
            out.println("/root/anaconda3/envs/codeBert/bin/python /data/Simfix/codebert/getVector.py --project_name=" + projectName + "--bug_id=" + bugId);
            //out.println("/Users/seventeen/opt/anaconda3/envs/codebert/bin/python getVector.py");
            out.println("exit");

            try {
                String line;
                while ((line = in.readLine()) != null) {
                    System.out.println(line);
                }
                proc.waitFor();
                in.close();
                out.close();
                proc.destroy();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        List<Pair<CodeBlock, Double>> tmpRes = new ArrayList<>();
        List<Pair<CodeBlock, Double>> sorted = new ArrayList<>();
        try {
            //BufferedReader in = new BufferedReader(new FileReader(HOME+"/clone/data_prediction/dict_result.csv"));
            BufferedReader in = new BufferedReader(new FileReader(base_url + "/similarity.txt"));
            String str;
            //String[] content = new String[3];
            for (Pair<CodeBlock, Double> codeBlockDoublePair : match) {
                str = in.readLine();

                if(Double.valueOf(str) > guard) {
                    tmpRes.add(new Pair<>(codeBlockDoublePair.getFirst(), Double.valueOf(str)));
                }
                // TODO:这个地方后期可以加入guard来限制次数
            }
            if(isFilter){
                for (Pair<CodeBlock, Double> candidatesPair : tmpRes) {
                    CodeBlock block = candidatesPair.getFirst();
                    if (_otherStruct.size() + _condStruct.size() > 0) {
                        if ((block.getCondStruct().size() + block.getOtherStruct().size()) == 0) {
                            continue;
                        }
                    }
                    Double similarity = candidatesPair.getSecond();

                    if (block.getCurrentLine() == 1 && _buggyCode.getCurrentLine() != 1) {
                        continue;
                    }
                    int i = 0;
                    boolean hasIntersection = false;
                    int replace = -1;
                    for (; i < sorted.size(); i++) {
                        Pair<CodeBlock, Double> pair = sorted.get(i);
                        if (pair.getFirst().hasIntersection(block)) {
                            hasIntersection = true;
                            if (similarity > pair.getSecond()) {
                                replace = i;
                            }
                            break;
                        }
                    }

                    if (hasIntersection) {
                        if (replace != -1) {
                            sorted.remove(replace);
                            sorted.add(new Pair<>(block, similarity));
                        }
                    } else {
                        sorted.add(new Pair<>(block, similarity));
                    }
                }
            } else {
                sorted = tmpRes;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        //排序
        Collections.sort(sorted, new Comparator<Pair<CodeBlock, Double>>() {
            @Override
            public int compare(Pair<CodeBlock, Double> o1, Pair<CodeBlock, Double> o2) {
                if(o1.getSecond() < o2.getSecond()){
                    return 1;
                } else if(o1.getSecond() > o2.getSecond()){
                    return -1;
                } else {
                    return 0;
                }
            }
        });
        System.out.println("Sorted candidate number: " + sorted.size());
        return sorted;
    }
    public List<Pair<CodeBlock, Double>> filter(String srcPath, double guard){
        List<String> files = JavaFile.ergodic(srcPath, new ArrayList<String>());
        List<Pair<CodeBlock, Double>> filtered = new ArrayList<>();
        CollectorVisitor collectorVisitor = new CollectorVisitor();
        int i=0;
        for(String file : files){
//			List<MethDecl> srcNode = JavaFiles.getNodes(file);
//				try {
//					File srcFile = new File("src.txt");
//					srcFile.createNewFile();
//					FileWriter fw = new FileWriter(srcFile.getAbsoluteFile());
//					BufferedWriter bw = new BufferedWriter(fw);
//					for(MethDecl oneNode: srcNode) {
//						bw.write(oneNode.toSrcString().toString());
//						bw.write("\n");
//					}
//					bw.close();
//					String[] cmdarray = {"cd /home/cyt/SimFix",""};
////					Process pro = Runtime.getRuntime().exec(cmdarray);
//					Process pro = Runtime.getRuntime().exec("mv /home/cyt/SimFix/src.txt /home/cyt");
//				} catch (IOException e) {
//					System.err.println("Failed to write file : " + file);
//				}
//			List<ASTNode> myast = new ArrayList<>();
//			for(MethDecl oneNode: srcNode){
//				myast.add(oneNode.getast());
//			}
//			CompilationUnit unit = JavaFiles.genAST(file);
//			CodeBlock buggyblock = new CodeBlock(file,unit, myast);
            CompilationUnit unit = null;
            try{
                unit = JavaFile.genASTFromFile(file);
            }catch(Exception e){
                File wrong = new File(file);
                wrong.delete();
                continue;
            }
            collectorVisitor.setUnit(file, unit);
//            if(i<6){
//                unit.accept(collectorVisitor);
//                i++;
//                continue;
//            }
            unit.accept(collectorVisitor);

            filtered = filter(filtered, guard);

        }

        Set<String> exist = new HashSet<>();//filter
        for(Pair<CodeBlock, Double> pair : filtered){
            if(exist.contains(pair.getFirst().toSrcString().toString())){
                continue;
            }
            exist.add(pair.getFirst().toSrcString().toString());
            double similarity = CodeBlockMatcher.getRewardSimilarity(_buggyCode, pair.getFirst()) + pair.getSecond();
            pair.setSecond(similarity);
        }

        Collections.sort(filtered, new Comparator<Pair<CodeBlock, Double>>() {
            @Override
            public int compare(Pair<CodeBlock, Double> o1, Pair<CodeBlock, Double> o2) {
                if(o1.getSecond() < o2.getSecond()){
                    return 1;
                } else if(o1.getSecond() > o2.getSecond()){
                    return -1;
                } else {
                    return 0;
                }
            }
        });

        return filtered;
    }

    private List<Pair<CodeBlock, Double>> filter(List<Pair<CodeBlock, Double>> filtered, double guard){
//		List<Pair<CodeBlock, Double>> filtered = new ArrayList<>();
        int delta = Constant.MAX_BLOCK_LINE - _buggyCode.getCurrentLine();
        delta = delta > 0 ? delta : 0;
        guard = guard + ((0.7 - guard) * delta / Constant.MAX_BLOCK_LINE ); // 0.9
//		System.out.println("Real guard value : " + guard);
        Set<String> codeRec = new HashSet<>();
        String ss = _buggyCode.toSrcString().toString();
        for(CodeBlock block : _candidates){
            if(_otherStruct.size() + _condStruct.size() > 0){
                if((block.getCondStruct().size() + block.getOtherStruct().size()) == 0){
                    continue;
                }
            }
            Double similarity = CodeBlockMatcher.getSimilarity(_buggyCode, block);
            if(similarity < guard){
//				System.out.println("Filtered by similiraty value : " + similarity);
                continue;
            }
//			similarity += CodeBlockMatcher.getRewardSimilarity(_buggyCode, block);
//			if (codeRec.contains(block.toSrcString().toString()) || _buggyCode.hasIntersection(block)) {
//				System.out.println("Duplicate >>>>>>>>>>>>>>>>");
//			} else {
            if(block.getCurrentLine() == 1 && _buggyCode.getCurrentLine() != 1){
                continue;
            }
            int i = 0;
            boolean hasIntersection = false;
            int replace = -1;
            for(; i < filtered.size(); i++){
                Pair<CodeBlock, Double> pair = filtered.get(i);
                if(pair.getFirst().hasIntersection(block)){
                    hasIntersection = true;
                    if(similarity > pair.getSecond()){
                        replace = i;
                    }
                    break;
                }
            }

            if(hasIntersection){
                if(replace != -1){
                    filtered.remove(replace);
                    codeRec.add(block.toSrcString().toString());
                    filtered.add(new Pair<CodeBlock, Double>(block, similarity));
                }
            } else {
                codeRec.add(block.toSrcString().toString());
                filtered.add(new Pair<CodeBlock, Double>(block, similarity));
            }
//			}
        }

        Collections.sort(filtered, new Comparator<Pair<CodeBlock, Double>>() {
            @Override
            public int compare(Pair<CodeBlock, Double> o1, Pair<CodeBlock, Double> o2) {
                if(o1.getSecond() < o2.getSecond()){
                    return 1;
                } else if(o1.getSecond() > o2.getSecond()){
                    return -1;
                } else {
                    return 0;
                }
            }
        });
        _candidates = new ArrayList<>();
        if(filtered.size() > 1000){
            for(int i = filtered.size() - 1; i > 1000; i--){
                filtered.remove(i);
            }
        }
        return filtered;
    }

    private List<Pair<CodeBlock, Double>> codefilter(List<Pair<CodeBlock, Double>> filtered) {
        for (CodeBlock block : _candidates) {
            if (_otherStruct.size() + _condStruct.size() > 0) {
                if ((block.getCondStruct().size() + block.getOtherStruct().size()) == 0) {
                    continue;
                }
            }
            if (block.getCurrentLine() == 1 && _buggyCode.getCurrentLine() != 1) {
                continue;
            }

            filtered.add(new Pair<CodeBlock, Double>(block, (double) 0));
        }
        _candidates = new ArrayList<>();
        if (filtered.size() > 1000) {
            for (int i = filtered.size() - 1; i > 1000; i--) {
                filtered.remove(i);
            }
        }
        return filtered;
    }

//    private List<Pair<CodeBlock, Double>> codefilter(List<Pair<CodeBlock, Double>> filtered){
//        for(CodeBlock block : _candidates){
//            if(_otherStruct.size() + _condStruct.size() > 0){
//                if((block.getCondStruct().size() + block.getOtherStruct().size()) == 0){
//                    continue;
//                }
//            }
//            if(block.getCurrentLine() == 1 && _buggyCode.getCurrentLine() != 1){
//                continue;
//            }
//            int i = 0;
//            boolean hasIntersection = false;
//            int replace = -1;
//            for(; i < filtered.size(); i++){
//                Pair<CodeBlock, Double> pair = filtered.get(i);
//                if(pair.getFirst().hasIntersection(block)){
//                    hasIntersection = true;
//                    break;
//                }
//            }
//
//            if(hasIntersection){
//                if(replace != -1){
//                    filtered.remove(replace);
//                    filtered.add(new Pair<CodeBlock, Double>(block, (double) 0));
//                }
//            } else {
//                filtered.add(new Pair<CodeBlock, Double>(block, (double) 0));
//            }
////			}
//        }
//        _candidates = new ArrayList<>();
//        if(filtered.size() > 1000){
//            for(int i = filtered.size() - 1; i > 1000; i--){
//                filtered.remove(i);
//            }
//        }
//        return filtered;
//    }

    class CollectorVisitor extends ASTVisitor{

        private CompilationUnit _unit = null;
        private String _fileName = null;

        public void setUnit(String fileName, CompilationUnit unit){
            _fileName = fileName;
            _unit = unit;
        }

        @Override
        public boolean visit(SimpleName node) {
            String name = node.getFullyQualifiedName();
            if(Character.isUpperCase(name.charAt(0))){
                return true;
            }
            Pair<String, String> classAndMethodName = NodeUtils.getTypeDecAndMethodDec(node);
            Type type = ProjectInfo.getVariableType(classAndMethodName.getFirst(), classAndMethodName.getSecond(), name);
            Variable variable = new Variable(null, name, type);
            boolean match = false;
            if(_variables.contains(variable) || _methods.contains(name) || sameStructure(node)){
                match = true;
            } else {
                ASTNode parent = node.getParent();
                while(parent != null && !(parent instanceof Statement)){
                    if(parent instanceof MethodInvocation){
                        if(_methods.contains(((MethodInvocation) parent).getName().getFullyQualifiedName())){
                            match = true;
                        }
                        break;
                    }
                    parent = parent.getParent();
                }
            }
            if(match){
                ASTNode parent = node.getParent();
                Statement statement = null;
                while(parent != null && !(parent instanceof MethodDeclaration)){
                    parent = parent.getParent();
                    if(statement == null && parent instanceof Statement){
                        statement = (Statement) parent;
                    }
                }
                // filter out anonymous classes
                if(parent != null && !(parent.getParent() instanceof AnonymousClassDeclaration)){
                    int line = _unit.getLineNumber(node.getStartPosition());
                    CodeSearch codeSearch = new CodeSearch(_unit, line, _buggyCode.getCurrentLine(), statement);
                    CodeBlock codeBlock = new CodeBlock(_fileName, _unit, codeSearch.getASTNodes());
//					CodeBlock codeBlock = new CodeBlock(_fileName, _unit, codeSearch.getASTNodes());
                    if(codeBlock.getCurrentLine() < _max_line){
                        _candidates.add(codeBlock);
                    }
                }
            }
            return true;
        }

        private boolean sameStructure(SimpleName name){
            return false;
//			if(_condStruct.size() == 0 && _otherStruct.size() == 0){
//				return false;
//			}
//			ASTNode parent = name.getParent();
//			Object kind = null;
//			while(parent != null){
//				if(parent instanceof MethodDeclaration){
//					break;
//				} else if(parent instanceof IfStatement){
//					kind = CondStruct.KIND.IF;
//					break;
//				} else if(parent instanceof SwitchStatement){
//					kind = CondStruct.KIND.SC;
//					break;
//				} else if(parent instanceof ReturnStatement){
//					kind = OtherStruct.KIND.RETURN;
//					break;
//				} else if(parent instanceof ConditionalExpression){
//					kind = CondStruct.KIND.CE;
//					break;
//				} else if(parent instanceof ThrowStatement){
//					kind = OtherStruct.KIND.THROW;
//					break;
//				}
//				parent = parent.getParent();
//			}
//			if(kind == null){
//				return false;
//			}
//			if(_condStruct.contains(kind) || _otherStruct.contains(kind)){
//				return true;
//			}
//			return false;
        }
    }

}